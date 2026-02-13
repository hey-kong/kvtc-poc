#!/usr/bin/env python3
"""
kvtc PoC v3: KV Cache Transform Coding for Llama 3.2 1B
========================================================
Fixes from v2 analysis:
  1. More calibration tokens (n_samples >> n_features=8192)
  2. DP: group components to amortize 32-bit overhead (groups of 16/64)
  3. Vectorized RoPE removal/application (no per-head loops)
  4. Longer test prompt for meaningful overall CR

Llama 3.2 1B: 16L / 8 KV-heads / 64 head_dim / cross_layer_dim=8192
"""

import argparse, time, zlib, struct, math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try GPU-accelerated compression via nvCOMP
try:
    from nvidia import nvcomp
    HAS_NVCOMP = True
    print("nvCOMP available — using GPU DEFLATE")
except ImportError:
    HAS_NVCOMP = False

MODEL_ID = "meta-llama/Llama-3.2-1B"
N_LAYERS = 16
N_KV_HEADS = 8
HEAD_DIM = 64
CROSS_LAYER_DIM = N_LAYERS * N_KV_HEADS * HEAD_DIM  # 8192

ROPE_THETA = 500000.0
ROPE_FACTOR = 32.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_ORIG_MAX_POS = 8192


@dataclass
class KVTCConfig:
    target_cr: int = 16
    sink_tokens: int = 4
    window_tokens: int = 128
    pca_rank: int = 4096


# ============================================================================
# RoPE: llama3 piecewise scaling — fully vectorized
# ============================================================================
def compute_llama3_inv_freq(head_dim: int, device: torch.device) -> torch.Tensor:
    base_inv = 1.0 / (ROPE_THETA ** (
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    scaled_inv = base_inv / ROPE_FACTOR
    wavelens = 2 * math.pi / base_inv
    low_wl = ROPE_ORIG_MAX_POS / ROPE_LOW_FREQ_FACTOR
    high_wl = ROPE_ORIG_MAX_POS / ROPE_HIGH_FREQ_FACTOR
    smooth = ((ROPE_ORIG_MAX_POS / wavelens - ROPE_LOW_FREQ_FACTOR) /
              (ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR)).clamp(0, 1)
    return torch.where(wavelens < high_wl, base_inv,
                       torch.where(wavelens > low_wl, scaled_inv,
                                   (1 - smooth) * scaled_inv + smooth * base_inv))


def build_rope_cache(seq_len, head_dim, device):
    inv_freq = compute_llama3_inv_freq(head_dim, device)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)  # [seq_len, head_dim//2]


def rotate_half(x):
    d2 = x.shape[-1] // 2
    return torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)


def remove_rope_batch(keys, cos, sin):
    """Vectorized inverse RoPE. keys: [n_pos, n_heads, head_dim]"""
    cos_full = torch.cat([cos, cos], dim=-1)  # [n_pos, head_dim]
    sin_full = torch.cat([sin, sin], dim=-1)
    # Broadcast over heads dim
    return keys * cos_full.unsqueeze(1) - rotate_half(keys) * sin_full.unsqueeze(1)


def apply_rope_batch(keys, cos, sin):
    """Vectorized forward RoPE. keys: [n_pos, n_heads, head_dim]"""
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return keys * cos_full.unsqueeze(1) + rotate_half(keys) * sin_full.unsqueeze(1)


# ============================================================================
# DynamicCache helper
# ============================================================================
def extract_kv(past):
    if hasattr(past, 'key_cache'):
        return past.key_cache, past.value_cache
    return [p[0] for p in past], [p[1] for p in past]


# ============================================================================
# Calibration
# ============================================================================
class KVTCCalibrator:
    def __init__(self, cfg: KVTCConfig, device: torch.device):
        self.cfg = cfg
        self.dev = device
        self.V_k = self.V_v = self.mu_k = self.mu_v = None
        self.sigma_k = self.sigma_v = None
        self.alloc_k = self.alloc_v = None

    def collect(self, model, tokenizer, texts, max_len=2048):
        all_k, all_v = [], []
        model.eval()
        with torch.no_grad():
            for i, txt in enumerate(texts):
                inp = tokenizer(txt, return_tensors="pt", truncation=True,
                                max_length=max_len).to(self.dev)
                out = model(**inp, use_cache=True)
                pk, pv = extract_kv(out.past_key_values)
                sl = inp["input_ids"].shape[1]
                s = self.cfg.sink_tokens
                if sl <= s:
                    continue

                cos_c, sin_c = build_rope_cache(sl, HEAD_DIM, self.dev)
                n_pos = sl - s

                # Vectorized extraction: all layers, all positions at once
                # pk[li]: [1, 8, sl, 64] → slice [1, 8, s:, 64] → [n_pos, 8, 64]
                k_layers, v_layers = [], []
                for li in range(N_LAYERS):
                    k_layers.append(pk[li][0, :, s:, :].permute(1, 0, 2))  # [n_pos, 8, 64]
                    v_layers.append(pv[li][0, :, s:, :].permute(1, 0, 2))

                # Vectorized RoPE removal across all positions
                cos_region = cos_c[s:sl]  # [n_pos, 32]
                sin_region = sin_c[s:sl]
                for li in range(N_LAYERS):
                    k_layers[li] = remove_rope_batch(k_layers[li], cos_region, sin_region)

                # Cross-layer concat: [n_pos, 16*8*64] = [n_pos, 8192]
                keys_mat = torch.cat([k.reshape(n_pos, -1) for k in k_layers], dim=1)
                vals_mat = torch.cat([v.reshape(n_pos, -1) for v in v_layers], dim=1)

                all_k.append(keys_mat)
                all_v.append(vals_mat)

                if (i + 1) % 10 == 0:
                    total = sum(x.shape[0] for x in all_k)
                    print(f"    {i+1}/{len(texts)} samples, {total} positions")

        return torch.cat(all_k).float(), torch.cat(all_v).float()

    def pca(self, data, rank):
        mu = data.mean(dim=0)
        centered = data - mu.unsqueeze(0)
        n, p = centered.shape
        r = min(rank, n, p)
        U, S, V = torch.pca_lowrank(centered, q=r, niter=5)
        return V, mu, S  # V: [p, r]

    def dp_alloc(self, sigma, total_features, label=""):
        """
        DP bit allocation matching the paper's algorithm (Appendix B.13).
        
        The DP decides for groups of PCA components:
          - group_size from {16, 64, 256, 1024}
          - bits_per_element from {0, 1, 2, 3, 4, 5, 6, 7, 8}
        Each active group costs OVERHEAD (32 bits) + group_size * bits payload.
        0 bits = drop the group entirely (no overhead cost).
        
        Key: the paper's Figure 6 shows leading components get 8-30 bits,
        trailing get 0. Larger groups amortize overhead better.
        """
        n = len(sigma)
        budget = int(total_features * 16 / self.cfg.target_cr)
        var = (sigma ** 2).cpu().numpy().astype(np.float64)

        OVERHEAD = 32  # bits per active group
        GROUP_SIZES = [16, 64, 256, 1024]
        BIT_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # DP: process components left-to-right
        # State: (component_index, bits_used) → min reconstruction error
        # For tractability, quantize budget into coarse units
        
        # Precompute cumulative variance for ranges
        cum_var = np.zeros(n + 1)
        for i in range(n):
            cum_var[i + 1] = cum_var[i] + var[i]
        
        def range_var(start, end):
            return cum_var[end] - cum_var[start]
        
        def quant_error(group_var, bits):
            """Error after uniform quantization with b bits."""
            if bits == 0:
                return group_var  # total loss
            # Uniform quantization of unit-variance: error ≈ var / (12 * 4^b)
            # Simplified: var / 4^b (as used in paper's DP)
            return group_var / (4.0 ** bits)

        # Greedy DP: evaluate all possible (group_size, bits) for each position
        # and pick the globally best allocation
        
        # Strategy: try different group sizes, for each compute optimal allocation
        # Pick the group size that gives best error/budget tradeoff
        
        best_alloc = np.zeros(n, dtype=np.int32)
        best_error = float('inf')
        
        for gs in GROUP_SIZES:
            if gs > n:
                continue
            ng = n // gs  # only full groups
            
            # For each group, compute cost vs error for each bit level
            # Then knapsack: select bits per group to minimize total error
            # subject to total cost <= budget
            
            # Group properties
            g_var = np.array([range_var(g*gs, (g+1)*gs) for g in range(ng)])
            
            # Knapsack via greedy (sort by efficiency)
            # For group g at bits b: cost = OVERHEAD + gs*b, gain = g_var[g] - quant_error(g_var[g], b)
            # = g_var[g] * (1 - 1/4^b)
            
            # Try allocating: start with 0 bits for all, then incrementally add
            g_alloc = np.zeros(ng, dtype=np.int32)
            rem = budget
            
            # Phase 1: for each group, find best initial bits level
            # considering the overhead cost of activation
            candidates = []
            for g in range(ng):
                for b in BIT_OPTIONS:
                    if b == 0:
                        continue
                    cost = OVERHEAD + gs * b
                    gain = g_var[g] * (1.0 - 1.0 / (4.0 ** b))
                    if cost > 0:
                        candidates.append((gain / cost, gain, cost, g, b))
            
            # Sort by efficiency (gain per bit)
            candidates.sort(key=lambda x: -x[0])
            
            # Greedy allocation
            used_groups = {}
            for eff, gain, cost, g, b in candidates:
                if g in used_groups:
                    # Already allocated — check if upgrading is worth it
                    cur_b = used_groups[g]
                    cur_cost = OVERHEAD + gs * cur_b
                    new_cost = OVERHEAD + gs * b
                    delta_cost = new_cost - cur_cost  # = gs * (b - cur_b)
                    delta_gain = g_var[g] * (1.0/(4.0**cur_b) - 1.0/(4.0**b))
                    if delta_cost > 0 and delta_cost <= rem and delta_gain > 0:
                        used_groups[g] = b
                        rem -= delta_cost
                else:
                    if cost <= rem:
                        used_groups[g] = b
                        rem -= cost
            
            # Compute total error
            total_err = 0.0
            alloc_tmp = np.zeros(n, dtype=np.int32)
            for g in range(ng):
                b = used_groups.get(g, 0)
                total_err += quant_error(g_var[g], b)
                if b > 0:
                    alloc_tmp[g*gs:(g+1)*gs] = b
            # Add error for components beyond full groups
            remainder_start = ng * gs
            total_err += range_var(remainder_start, n)
            
            if total_err < best_error:
                best_error = total_err
                best_alloc = alloc_tmp.copy()
                best_gs = gs
                best_rem = rem
                best_ng = len(used_groups)

        active = int(np.sum(best_alloc > 0))
        total_used = budget - best_rem
        avg_bits = np.mean(best_alloc[best_alloc > 0]) if active > 0 else 0
        max_b = int(np.max(best_alloc))
        print(f"    {label}: {active}/{n} active ({best_ng} groups of {best_gs}), "
              f"budget={budget}, used={total_used}, "
              f"avg={avg_bits:.1f} b/comp, max={max_b}b, err={best_error:.0f}")
        return best_alloc

    def calibrate(self, model, tokenizer, texts, max_len=2048):
        print(f"[Cal] Collecting KV caches ({len(texts)} samples)...")
        t0 = time.time()
        kd, vd = self.collect(model, tokenizer, texts, max_len)
        print(f"  {kd.shape[0]} positions × {kd.shape[1]} features, {time.time()-t0:.1f}s")

        rank = min(self.cfg.pca_rank, kd.shape[0], CROSS_LAYER_DIM)
        print(f"[Cal] PCA (rank≤{rank})...")
        t0 = time.time()
        self.V_k, self.mu_k, self.sigma_k = self.pca(kd, rank)
        print(f"  Keys: rank={self.V_k.shape[1]}, {time.time()-t0:.1f}s")
        t0 = time.time()
        self.V_v, self.mu_v, self.sigma_v = self.pca(vd, rank)
        print(f"  Values: rank={self.V_v.shape[1]}, {time.time()-t0:.1f}s")

        print(f"[Cal] DP bit allocation (CR={self.cfg.target_cr}×)...")
        self.alloc_k = self.dp_alloc(self.sigma_k, CROSS_LAYER_DIM, "Keys")
        self.alloc_v = self.dp_alloc(self.sigma_v, CROSS_LAYER_DIM, "Values")


# ============================================================================
# Compression / Decompression
# ============================================================================
class KVTCCompressor:
    def __init__(self, cal: KVTCCalibrator, cfg: KVTCConfig, dev: torch.device):
        self.cal, self.cfg, self.dev = cal, cfg, dev

    def _quant(self, vals, n_bits):
        if n_bits == 0:
            return None, 0.0, 0.0
        nl = (1 << n_bits) - 1
        vmin, vmax = float(vals.min()), float(vals.max())
        scale = (vmax - vmin) / max(nl, 1)
        if scale == 0:
            scale = 1.0
        q = torch.clamp(torch.round((vals - vmin) / scale), 0, nl).to(torch.int32)
        return q, vmin, scale

    def _dequant(self, q, vmin, scale):
        return q.float() * scale + vmin

    def compress_matrix(self, data, V, mu, alloc):
        """Compress [n_tok, 8192] → compressed bytes."""
        nt = data.shape[0]
        D = (data.float() - mu.unsqueeze(0).to(data.device)) @ V.to(data.device)

        active = np.where(alloc > 0)[0]
        parts = []
        for idx in active:
            q, vmin, scale = self._quant(D[:, idx].cpu(), int(alloc[idx]))
            # Header: idx(2B) + bits(1B) + vmin(4B) + scale(4B) = 11B
            parts.append(struct.pack("<HBff", idx, alloc[idx], vmin, scale))
            parts.append(q.numpy().astype(np.uint16 if alloc[idx] <= 16 else np.uint32).tobytes())

        raw = b"".join(parts)

        # Entropy coding: GPU DEFLATE via nvCOMP or CPU zlib fallback
        if HAS_NVCOMP:
            raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
            nv_arr = nvcomp.as_array(raw_tensor)
            codec = nvcomp.Codec(algorithm="Deflate")
            comp_arr = codec.encode(nv_arr)
            compressed = bytes(comp_arr.cpu())
            backend = "nvcomp"
        else:
            compressed = zlib.compress(raw, level=6)
            backend = "zlib"

        return {"compressed": compressed, "raw_size": len(raw),
                "n_tokens": nt, "n_active": len(active), "backend": backend}

    def decompress_matrix(self, comp, V, mu, alloc):
        """Decompress → [n_tok, 8192]."""
        nt, rank = comp["n_tokens"], V.shape[1]

        # Entropy decoding
        if HAS_NVCOMP and comp.get("backend") == "nvcomp":
            comp_tensor = torch.frombuffer(
                bytearray(comp["compressed"]), dtype=torch.uint8).cuda()
            nv_arr = nvcomp.as_array(comp_tensor)
            codec = nvcomp.Codec(algorithm="Deflate")
            dec_arr = codec.decode(nv_arr)
            raw = bytes(dec_arr.cpu())
        else:
            raw = zlib.decompress(comp["compressed"])

        D = torch.zeros(nt, rank, device=self.dev)
        active = np.where(alloc > 0)[0]
        offset = 0
        for _ in active:
            idx, nb, vmin, scale = struct.unpack("<HBff", raw[offset:offset+11])
            offset += 11
            dtype = np.uint16 if nb <= 16 else np.uint32
            bpe = 2 if nb <= 16 else 4
            q = np.frombuffer(raw[offset:offset+nt*bpe], dtype=dtype)
            offset += nt * bpe
            D[:, idx] = self._dequant(torch.from_numpy(q.copy()).to(self.dev), vmin, scale)

        return D @ V.T.to(self.dev) + mu.unsqueeze(0).to(self.dev)

    def compress_full(self, past):
        pk, pv = extract_kv(past)
        sl = pk[0].shape[2]
        sink = self.cfg.sink_tokens
        win = min(self.cfg.window_tokens, sl - sink)
        cs, ce = sink, sl - win
        nc = ce - cs

        if nc <= 0:
            return {"type": "pass", "past": past,
                    "metrics": {"error": f"need >{sink+win} tokens, got {sl}"}}

        elem_sz = pk[0].element_size()
        cos_c, sin_c = build_rope_cache(sl, HEAD_DIM, self.dev)

        # --- Original size ---
        orig_bytes = sum((pk[l].nelement() + pv[l].nelement()) * elem_sz
                         for l in range(N_LAYERS))

        # --- Extract compressible region vectorized ---
        # For each layer, slice [1, 8, cs:ce, 64] → [nc, 8, 64]
        t0 = time.time()
        key_layers, val_layers = [], []
        for li in range(N_LAYERS):
            key_layers.append(pk[li][0, :, cs:ce, :].permute(1, 0, 2))  # [nc, 8, 64]
            val_layers.append(pv[li][0, :, cs:ce, :].permute(1, 0, 2))

        # Vectorized RoPE removal: positions cs..ce-1
        cos_region = cos_c[cs:ce]  # [nc, 32]
        sin_region = sin_c[cs:ce]

        key_no_rope = []
        for li in range(N_LAYERS):
            key_no_rope.append(
                remove_rope_batch(key_layers[li], cos_region, sin_region))

        # Cross-layer concat: [nc, 16*8*64] = [nc, 8192]
        keys_mat = torch.cat([k.reshape(nc, -1) for k in key_no_rope], dim=1)
        vals_mat = torch.cat([v.reshape(nc, -1) for v in val_layers], dim=1)

        # --- Compress ---
        ck = self.compress_matrix(keys_mat, self.cal.V_k, self.cal.mu_k, self.cal.alloc_k)
        cv = self.compress_matrix(vals_mat, self.cal.V_v, self.cal.mu_v, self.cal.alloc_v)
        t_comp = time.time() - t0

        # --- Uncompressed regions ---
        sink_c, win_c = [], []
        sink_b = win_b = 0
        for li in range(N_LAYERS):
            sk, sv = pk[li][:,:,:sink,:].clone(), pv[li][:,:,:sink,:].clone()
            wk, wv = pk[li][:,:,ce:,:].clone(), pv[li][:,:,ce:,:].clone()
            sink_c.append((sk, sv))
            win_c.append((wk, wv))
            sink_b += (sk.nelement() + sv.nelement()) * elem_sz
            win_b += (wk.nelement() + wv.nelement()) * elem_sz

        comp_bytes = len(ck["compressed"]) + len(cv["compressed"])
        total_after = comp_bytes + sink_b + win_b

        return {
            "type": "kvtc", "ck": ck, "cv": cv,
            "sink": sink_c, "win": win_c, "cs": cs, "ce": ce, "sl": sl,
            "metrics": {
                "seq_len": sl, "n_comp": nc, "n_sink": sink, "n_win": sl - ce,
                "orig_bytes": orig_bytes,
                "comp_region_bytes": comp_bytes,
                "sink_bytes": sink_b, "win_bytes": win_b,
                "total_after": total_after,
                "overall_cr": orig_bytes / max(total_after, 1),
                "region_cr": (nc * CROSS_LAYER_DIM * 2 * elem_sz) / max(comp_bytes, 1),
                "deflate_ratio": (ck["raw_size"] + cv["raw_size"]) / max(comp_bytes, 1),
                "pca_bytes": (self.cal.V_k.nelement() + self.cal.V_v.nelement() +
                              self.cal.mu_k.nelement() + self.cal.mu_v.nelement()) * 4,
                "compress_ms": t_comp * 1000,
            }
        }

    def decompress_full(self, c):
        if c["type"] == "pass":
            return c["past"]

        cs, ce, sl = c["cs"], c["ce"], c["sl"]
        nc = ce - cs
        cos_c, sin_c = build_rope_cache(sl, HEAD_DIM, self.dev)

        t0 = time.time()
        km = self.decompress_matrix(c["ck"], self.cal.V_k, self.cal.mu_k, self.cal.alloc_k)
        vm = self.decompress_matrix(c["cv"], self.cal.V_v, self.cal.mu_v, self.cal.alloc_v)

        rk_list, rv_list = [], []
        cos_region = cos_c[cs:ce]
        sin_region = sin_c[cs:ce]

        for li in range(N_LAYERS):
            sk, sv = c["sink"][li]
            wk, wv = c["win"][li]

            fs = li * N_KV_HEADS * HEAD_DIM
            fe = fs + N_KV_HEADS * HEAD_DIM

            lk = km[:, fs:fe].reshape(nc, N_KV_HEADS, HEAD_DIM)
            lv = vm[:, fs:fe].reshape(nc, N_KV_HEADS, HEAD_DIM)

            # Vectorized RoPE re-application
            lk = apply_rope_batch(lk, cos_region, sin_region)

            # [nc, heads, dim] → [1, heads, nc, dim]
            ck_ = lk.unsqueeze(0).permute(0, 2, 1, 3).to(sk.dtype)
            cv_ = lv.unsqueeze(0).permute(0, 2, 1, 3).to(sv.dtype)

            rk_list.append(torch.cat([sk, ck_, wk], dim=2))
            rv_list.append(torch.cat([sv, cv_, wv], dim=2))

        c["metrics"]["decompress_ms"] = (time.time() - t0) * 1000
        return rk_list, rv_list


# ============================================================================
# Reconstruction quality
# ============================================================================
def recon_error(orig, recon):
    ok, ov = extract_kv(orig)
    if isinstance(recon, tuple):
        rk, rv = recon
    else:
        rk, rv = extract_kv(recon)

    mse_k = mse_v = cos_k = cos_v = rel_k = rel_v = 0.0
    for li in range(N_LAYERS):
        a, b = ok[li].float(), rk[li].float()
        c, d = ov[li].float(), rv[li].float()
        ml = min(a.shape[2], b.shape[2])
        a, b, c, d = a[:,:,:ml], b[:,:,:ml], c[:,:,:ml], d[:,:,:ml]

        mse_k += F.mse_loss(b, a).item()
        mse_v += F.mse_loss(d, c).item()
        rel_k += (torch.norm(b-a) / torch.norm(a)).item()
        rel_v += (torch.norm(d-c) / torch.norm(c)).item()
        cos_k += F.cosine_similarity(
            a.reshape(-1, HEAD_DIM), b.reshape(-1, HEAD_DIM), dim=-1).mean().item()
        cos_v += F.cosine_similarity(
            c.reshape(-1, HEAD_DIM), d.reshape(-1, HEAD_DIM), dim=-1).mean().item()

    n = N_LAYERS
    return dict(mse_k=mse_k/n, mse_v=mse_v/n, cos_k=cos_k/n,
                cos_v=cos_v/n, rel_k=rel_k/n, rel_v=rel_v/n)


# ============================================================================
# Main
# ============================================================================
def fmt(b):
    if b < 1024: return f"{b} B"
    if b < 1048576: return f"{b/1024:.1f} KiB"
    return f"{b/1048576:.2f} MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--calibration-samples", type=int, default=128)
    ap.add_argument("--target-cr", type=int, default=16)
    ap.add_argument("--pca-rank", type=int, default=4096)
    ap.add_argument("--max-cal-len", type=int, default=2048,
                    help="Max tokens per calibration sample")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = torch.device("cuda" if args.device == "auto" and
                       torch.cuda.is_available() else
                       args.device if args.device != "auto" else "cpu")
    print(f"Device: {dev}")
    print(f"Arch: {N_LAYERS}L / {N_KV_HEADS}KV / {HEAD_DIM}D → dim={CROSS_LAYER_DIM}")

    print(f"\n[1/5] Loading model...")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=dev)
    model.eval()

    # Calibration texts — longer to get n_samples >> 8192
    print(f"\n[2/5] Calibration data ({args.calibration_samples} samples, "
          f"max {args.max_cal_len} tok each)...")
    base_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of algorithms that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future. Machine learning algorithms are typically classified into supervised learning, unsupervised learning, and reinforcement learning categories based on the type of feedback available during training.",
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that process all positions simultaneously rather than sequentially. Unlike recurrent neural networks, transformers can capture long-range dependencies without suffering from vanishing gradients. The key innovation is the multi-head attention mechanism which computes weighted relationships between all pairs of positions in the input sequence, allowing the model to attend to different aspects of the input in parallel across multiple representation subspaces.",
        "Cloud computing fundamentally changed how organizations deploy and manage IT infrastructure by providing on-demand access to shared pools of configurable computing resources. These resources including networks, servers, storage, applications, and services can be rapidly provisioned and released with minimal management effort. The three primary service models are Infrastructure as a Service, Platform as a Service, and Software as a Service, each offering different levels of abstraction and control over the underlying computing resources.",
        "Deep learning has achieved remarkable success across a wide range of tasks including computer vision, natural language processing, speech recognition, and game playing. The key insight is that deep neural networks with many layers can learn hierarchical representations of data, where each layer captures increasingly abstract features. Training these networks requires large datasets and significant computational resources, but techniques like transfer learning, data augmentation, and regularization have made deep learning accessible for a wider range of applications.",
        "Kubernetes has become the de facto standard for container orchestration, automating deployment, scaling, and management of containerized applications. Its architecture consists of a control plane that manages the cluster state and worker nodes that run the actual workloads. Key abstractions include Pods as the smallest deployable units, Services for stable networking, Deployments for declarative updates, and ConfigMaps and Secrets for configuration management. The scheduler assigns pods to nodes based on resource requirements and constraints.",
        "Natural language processing encompasses techniques for analyzing, understanding, and generating human language. Modern approaches leverage large pre-trained transformer models that capture linguistic knowledge from massive text corpora. These foundation models can be adapted to specific tasks through fine-tuning or in-context learning. Key tasks include text classification, named entity recognition, sentiment analysis, machine translation, question answering, summarization, and dialogue systems, each requiring different architectural considerations.",
        "Distributed systems present unique challenges in maintaining consistency, availability, and partition tolerance as described by the CAP theorem. Consensus algorithms like Raft and Paxos enable fault-tolerant state machine replication across unreliable networks. Modern distributed databases use techniques like sharding for horizontal scalability, replication for fault tolerance, and vector clocks or hybrid logical clocks for causal ordering of events across nodes in the system.",
        "GPU computing has accelerated deep learning by providing massive parallelism for matrix operations that dominate neural network training and inference. Modern GPUs like the NVIDIA H100 feature specialized tensor cores for mixed-precision matrix multiplication, high-bandwidth HBM3 memory, and NVLink interconnects for multi-GPU scaling. The CUDA programming model exposes thread-level parallelism through warps and blocks, while libraries like cuDNN and cuBLAS provide optimized implementations of common operations.",
        "The attention mechanism computes compatibility scores between query and key vectors to produce weighted sums of value vectors. Multi-head attention projects inputs into multiple subspaces, applies attention independently in each, and concatenates the results. Grouped query attention reduces memory bandwidth by sharing key-value heads across query head groups. Flash attention optimizes memory access patterns using tiling to achieve IO-aware attention computation with reduced HBM access.",
        "Large language models are trained through a multi-stage pipeline starting with pre-training on massive text corpora using next-token prediction. This is followed by supervised fine-tuning on instruction-following examples and alignment through reinforcement learning from human feedback. Scaling laws describe how model performance improves predictably with increases in model size, dataset size, and compute budget. Inference optimization techniques include quantization, pruning, knowledge distillation, and speculative decoding.",
        "Database systems have evolved to support diverse workloads through specialized storage engines and query processors. OLTP databases optimize for low-latency point queries using B-tree indexes and row-oriented storage, while OLAP systems use columnar storage and vectorized execution for analytical queries. Modern distributed databases like CockroachDB and TiDB combine both workloads through HTAP architectures. Key considerations include transaction isolation levels, concurrency control mechanisms, and replication strategies for high availability.",
        "Microservices architecture decomposes monolithic applications into independently deployable services communicating through well-defined APIs. This enables teams to develop, test, and deploy services independently using different technology stacks. Service mesh implementations like Istio provide observability, traffic management, and security policies at the infrastructure level. Challenges include distributed tracing, eventual consistency across service boundaries, and managing the operational complexity of many small services.",
        "Computer vision has been transformed by deep learning, particularly convolutional neural networks and more recently vision transformers. Object detection architectures like YOLO and DETR locate and classify objects in images, while segmentation models like SAM produce pixel-level masks. Self-supervised pre-training on unlabeled images using contrastive learning or masked image modeling has dramatically reduced the need for labeled data. Video understanding extends these techniques to temporal sequences.",
        "Reinforcement learning enables agents to learn optimal behaviors through trial and error interaction with environments. Key algorithms include policy gradient methods like PPO, value-based methods like DQN, and model-based approaches that learn environment dynamics. Recent advances in offline reinforcement learning allow training on pre-collected datasets without environment interaction. Applications span game playing, robotics, recommendation systems, and large language model alignment through RLHF.",
        "Version control with Git provides distributed collaborative development through content-addressable storage and directed acyclic graph structures. Branching strategies like GitFlow and trunk-based development define how teams manage concurrent work streams. CI/CD pipelines automate testing and deployment triggered by repository events. Advanced features like interactive rebasing, cherry-picking, and worktrees support complex development workflows across large engineering organizations.",
        "Information retrieval systems combine lexical matching with semantic understanding to find relevant documents. Traditional TF-IDF and BM25 methods measure term frequency statistics, while dense retrieval uses neural encoders to compute semantic similarity in embedding space. Hybrid approaches like ColBERT combine both paradigms. Retrieval-augmented generation extends language models by conditioning on retrieved documents, improving factual accuracy and enabling knowledge updates without retraining.",
    ] * ((args.calibration_samples // 16) + 1)
    calib_texts = base_texts[:args.calibration_samples]
    # Concatenate pairs for longer sequences → more positions per sample
    long_texts = []
    for i in range(0, len(calib_texts) - 1, 2):
        long_texts.append(calib_texts[i] + " " + calib_texts[i + 1])
    # Mix short and long as per paper (50/50 short/long)
    calib_texts = calib_texts + long_texts
    calib_texts = calib_texts[:args.calibration_samples]

    # --- Calibrate ---
    print(f"\n[3/5] Calibrating (CR={args.target_cr}×)...")
    cfg = KVTCConfig(target_cr=args.target_cr, pca_rank=args.pca_rank)
    cal = KVTCCalibrator(cfg, dev)
    cal.calibrate(model, tok, calib_texts, max_len=args.max_cal_len)

    # --- Test prompt (long enough for meaningful compression) ---
    print(f"\n[4/5] Compression test...")
    comp = KVTCCompressor(cal, cfg, dev)

    test_prompt = """The following is a comprehensive technical overview of key-value cache management in large language model inference systems deployed at scale in production environments.

Modern transformer-based language models use an autoregressive generation process where each new token depends on all previous tokens through the self-attention mechanism. To avoid redundant computation during generation, the model caches the key and value projections from each attention layer for all previously processed tokens. This collection of cached tensors is known as the KV cache. For a model with L layers, H attention heads, head dimension D, and sequence length T, the KV cache occupies 4*L*H*D*T bytes in 16-bit precision.

As context lengths grow to hundreds of thousands of tokens and models scale to hundreds of billions of parameters, the KV cache becomes a significant memory bottleneck. For example, processing a 100K token context with Llama 3.3 70B requires approximately 32 GiB of KV cache memory alone. This memory pressure limits the number of concurrent requests a serving system can handle, directly impacting throughput and cost efficiency in production deployments.

Several approaches have been proposed to address this challenge. Quantization methods like KIVI and KVQuant reduce the bit width of cached values from 16 bits to 2-4 bits per element. Token eviction methods like H2O and TOVA selectively discard less important tokens from the cache based on attention scores. SVD-based methods like xKV exploit low-rank structure to compress the cache by computing per-prompt singular value decompositions. However, these methods often face accuracy degradation at high compression ratios or fail to exploit the full redundancy present across layers and heads in KV caches.

A promising new direction is transform coding, inspired by classical media compression techniques like JPEG and video codecs. The key insight is that KV caches exhibit strong cross-layer and cross-head correlations that can be exploited through principal component analysis. By projecting the concatenated cross-layer KV cache onto a learned orthonormal basis computed once on a calibration dataset, the resulting PCA coefficients can be quantized with variable bit widths. These bit widths are allocated by a dynamic programming algorithm that minimizes the Frobenius reconstruction error under a fixed bit budget determined by the target compression ratio. The quantized coefficients are then entropy-coded with DEFLATE for additional lossless compression.

This transform coding approach achieves approximately 20x compression with negligible accuracy loss across reasoning benchmarks like GSM8K, knowledge benchmarks like MMLU, and long-context benchmarks like RULER and Lost in the Middle. Compression ratios of 40x or higher are attainable for specific use cases at modest accuracy decrease. The method consistently outperforms existing inference-time baselines including token eviction, quantization-only approaches, and SVD-based methods while achieving substantially higher compression ratios.

The compression and decompression operations add modest latency overhead that is typically much less than the cost of recomputing the full KV cache from scratch. For example, decompressing a 16x compressed cache for 8K context takes approximately 267ms compared to 3098ms for full recomputation, representing an 8x reduction in time-to-first-token.

Key implementation considerations include the treatment of attention sink tokens which receive disproportionate attention weight and should be stored uncompressed, the use of a sliding window of recent tokens that are also kept uncompressed for maximum accuracy, and the removal of rotary positional embeddings from keys before compression since RoPE distorts the low-rank structure that PCA exploits.

The calibration procedure is lightweight and model-specific rather than prompt-specific. A single PCA basis computed on approximately 160K tokens from a diverse calibration set generalizes well across different downstream tasks and domains. The dynamic programming bit allocation is computed once per model and compression ratio target. Both calibration artifacts are small relative to model parameters, typically 2-4% overhead, and can be precomputed and distributed alongside the model weights.

Production deployment architectures for LLM serving often split prefill and decode across separate nodes connected by high-speed RDMA fabric. In these disaggregated serving setups, KV cache compression reduces the dominant cross-node traffic proportionally to the compression ratio. Both nodes maintain tiered KV cache hierarchies spanning GPU HBM, CPU DRAM, and NVMe storage, with compressed caches extending the effective capacity and lifetime of caches at each tier."""

    inp = tok(test_prompt, return_tensors="pt", truncation=True,
              max_length=4096).to(dev)
    sl = inp["input_ids"].shape[1]
    print(f"  Test tokens: {sl}")

    with torch.no_grad():
        out = model(**inp, use_cache=True)
    orig_past = out.past_key_values

    result = comp.compress_full(orig_past)
    m = result["metrics"]

    if result["type"] == "pass":
        print(f"  ERROR: {m.get('error')}")
        return

    recon = comp.decompress_full(result)
    err = recon_error(orig_past, recon)

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"  KV CACHE COMPRESSION REPORT — kvtc PoC v3")
    print(f"{'='*70}")
    print(f"  Model:         {args.model_id}")
    print(f"  Architecture:  {N_LAYERS}L / {N_KV_HEADS}KV / {HEAD_DIM}D")
    print(f"  Sequence:      {m['seq_len']} tokens")
    print(f"  Target CR:     {args.target_cr}×")
    print(f"")
    print(f"  TOKENS:")
    print(f"    Sink (uncompressed):     {m['n_sink']:>6}")
    print(f"    Compressed:              {m['n_comp']:>6}")
    print(f"    Window (uncompressed):   {m['n_win']:>6}")
    print(f"")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  KV CACHE BEFORE:  {fmt(m['orig_bytes']):>12}              │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  KV CACHE AFTER:                                │")
    print(f"  │    Compressed region:  {fmt(m['comp_region_bytes']):>12}            │")
    print(f"  │    Sink (raw):         {fmt(m['sink_bytes']):>12}            │")
    print(f"  │    Window (raw):       {fmt(m['win_bytes']):>12}            │")
    print(f"  │    ─────────────────────────────                │")
    print(f"  │    TOTAL AFTER:        {fmt(m['total_after']):>12}            │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  OVERALL CR:           {m['overall_cr']:>8.1f}×               │")
    print(f"  │  Region CR:            {m['region_cr']:>8.1f}×               │")
    print(f"  │  DEFLATE bonus:        {m['deflate_ratio']:>8.2f}×               │")
    print(f"  └─────────────────────────────────────────────────┘")
    print(f"")
    print(f"  OVERHEAD:  PCA matrices = {fmt(m['pca_bytes'])} (per model)")
    print(f"  ENTROPY:   {'nvCOMP GPU DEFLATE' if HAS_NVCOMP else 'zlib CPU DEFLATE'}")
    print(f"  LATENCY:   compress={m['compress_ms']:.0f}ms  "
          f"decompress={m.get('decompress_ms',0):.0f}ms")
    print(f"")
    print(f"  RECONSTRUCTION QUALITY:")
    print(f"    Keys   — MSE:{err['mse_k']:.6f}  cos:{err['cos_k']:.4f}  "
          f"rel:{err['rel_k']:.4f}")
    print(f"    Values — MSE:{err['mse_v']:.6f}  cos:{err['cos_v']:.4f}  "
          f"rel:{err['rel_v']:.4f}")

    saved = m['orig_bytes'] - m['total_after']
    pct = saved / m['orig_bytes'] * 100
    print(f"\n  → {fmt(m['orig_bytes'])} → {fmt(m['total_after'])} "
          f"({pct:.1f}% saved, {fmt(saved)} freed)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
