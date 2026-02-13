#!/usr/bin/env python3
"""
kvtc RAG PoC: Multi-turn RAG with KV Cache Compression
=======================================================
Simulates a real RAG workflow:
  1. Prefill a long context (system prompt + retrieved documents)
  2. User asks questions about the documents
  3. Between turns, compress the KV cache to free GPU memory
  4. On next question, decompress instead of recomputing

Compares three strategies:
  A) Recompute: discard cache, full prefill every turn
  B) Hold: keep cache in HBM (fast but wastes memory)
  C) kvtc: compress → store → decompress between turns

Usage:
  python kvtc_rag_poc.py
  python kvtc_rag_poc.py --target-cr 8 --num-turns 5
"""

import argparse, time, zlib, struct, math, gc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from nvidia import nvcomp
    HAS_NVCOMP = True
except ImportError:
    HAS_NVCOMP = False

# ============================================================================
# Model config — Llama 3.2 1B
# ============================================================================
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
# RoPE (llama3 piecewise scaling)
# ============================================================================
def compute_llama3_inv_freq(hd, dev):
    base = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    scaled = base / ROPE_FACTOR
    wl = 2 * math.pi / base
    lo_wl, hi_wl = ROPE_ORIG_MAX_POS / ROPE_LOW_FREQ_FACTOR, ROPE_ORIG_MAX_POS / ROPE_HIGH_FREQ_FACTOR
    sm = ((ROPE_ORIG_MAX_POS / wl - ROPE_LOW_FREQ_FACTOR) /
          (ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR)).clamp(0, 1)
    return torch.where(wl < hi_wl, base, torch.where(wl > lo_wl, scaled, (1-sm)*scaled + sm*base))


def build_rope_cache(sl, hd, dev):
    inv = compute_llama3_inv_freq(hd, dev)
    t = torch.arange(sl, device=dev, dtype=torch.float32)
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def rotate_half(x):
    d2 = x.shape[-1] // 2
    return torch.cat([-x[..., d2:], x[..., :d2]], dim=-1)


def remove_rope_batch(k, cos, sin):
    cf, sf = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    return k * cf.unsqueeze(1) - rotate_half(k) * sf.unsqueeze(1)


def apply_rope_batch(k, cos, sin):
    cf, sf = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    return k * cf.unsqueeze(1) + rotate_half(k) * sf.unsqueeze(1)


# ============================================================================
# DynamicCache helper
# ============================================================================
def extract_kv(past):
    if hasattr(past, 'key_cache'):
        return past.key_cache, past.value_cache
    return [p[0] for p in past], [p[1] for p in past]


def kv_memory_bytes(past):
    pk, pv = extract_kv(past)
    total = 0
    for li in range(len(pk)):
        total += pk[li].nelement() * pk[li].element_size()
        total += pv[li].nelement() * pv[li].element_size()
    return total


# ============================================================================
# Calibrator (reused from kvtc_poc.py, simplified)
# ============================================================================
class KVTCCalibrator:
    def __init__(self, cfg, dev):
        self.cfg, self.dev = cfg, dev
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
                k_layers, v_layers = [], []
                for li in range(N_LAYERS):
                    k_layers.append(pk[li][0, :, s:, :].permute(1, 0, 2))
                    v_layers.append(pv[li][0, :, s:, :].permute(1, 0, 2))
                cos_r, sin_r = cos_c[s:sl], sin_c[s:sl]
                for li in range(N_LAYERS):
                    k_layers[li] = remove_rope_batch(k_layers[li], cos_r, sin_r)
                km = torch.cat([k.reshape(n_pos, -1) for k in k_layers], dim=1)
                vm = torch.cat([v.reshape(n_pos, -1) for v in v_layers], dim=1)
                all_k.append(km)
                all_v.append(vm)
        return torch.cat(all_k).float(), torch.cat(all_v).float()

    def pca(self, data, rank):
        mu = data.mean(dim=0)
        centered = data - mu.unsqueeze(0)
        r = min(rank, centered.shape[0], centered.shape[1])
        U, S, V = torch.pca_lowrank(centered, q=r, niter=5)
        return V, mu, S

    def dp_alloc(self, sigma, total_features, label=""):
        n = len(sigma)
        budget = int(total_features * 16 / self.cfg.target_cr)
        var = (sigma ** 2).cpu().numpy().astype(np.float64)
        OVERHEAD, GROUP_SIZES = 32, [16, 64, 256, 1024]
        BIT_OPTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        cum = np.zeros(n + 1)
        for i in range(n):
            cum[i+1] = cum[i] + var[i]

        best_alloc, best_err = np.zeros(n, dtype=np.int32), float('inf')
        best_gs, best_ng = 16, 0
        for gs in GROUP_SIZES:
            if gs > n: continue
            ng = n // gs
            gv = np.array([cum[(g+1)*gs] - cum[g*gs] for g in range(ng)])
            cands = []
            for g in range(ng):
                for b in BIT_OPTS:
                    if b == 0: continue
                    cost = OVERHEAD + gs * b
                    gain = gv[g] * (1.0 - 1.0 / (4.0 ** b))
                    if cost > 0:
                        cands.append((gain / cost, gain, cost, g, b))
            cands.sort(key=lambda x: -x[0])
            used, rem = {}, budget
            for _, gain, cost, g, b in cands:
                if g in used:
                    cur_b = used[g]
                    dc = gs * (b - cur_b)
                    dg = gv[g] * (1.0/(4.0**cur_b) - 1.0/(4.0**b))
                    if dc > 0 and dc <= rem and dg > 0:
                        used[g] = b; rem -= dc
                else:
                    if cost <= rem:
                        used[g] = b; rem -= cost
            err = 0.0
            alloc_tmp = np.zeros(n, dtype=np.int32)
            for g in range(ng):
                b = used.get(g, 0)
                err += gv[g] / (4.0**b) if b > 0 else gv[g]
                if b > 0: alloc_tmp[g*gs:(g+1)*gs] = b
            err += cum[n] - cum[ng*gs]
            if err < best_err:
                best_err, best_alloc = err, alloc_tmp.copy()
                best_gs, best_ng = gs, len(used)

        active = int(np.sum(best_alloc > 0))
        avg = np.mean(best_alloc[best_alloc > 0]) if active > 0 else 0
        print(f"    {label}: {active}/{n} active ({best_ng} groups of {best_gs}), "
              f"avg={avg:.1f}b, max={int(np.max(best_alloc))}b")
        return best_alloc

    def calibrate(self, model, tokenizer, texts, max_len=2048):
        print(f"  Collecting KV caches ({len(texts)} samples)...")
        t0 = time.time()
        kd, vd = self.collect(model, tokenizer, texts, max_len)
        print(f"  {kd.shape[0]} positions × {kd.shape[1]} dim, {time.time()-t0:.1f}s")
        rank = min(self.cfg.pca_rank, kd.shape[0], CROSS_LAYER_DIM)
        print(f"  PCA (rank≤{rank})...")
        self.V_k, self.mu_k, self.sigma_k = self.pca(kd, rank)
        self.V_v, self.mu_v, self.sigma_v = self.pca(vd, rank)
        print(f"  DP bit allocation (CR={self.cfg.target_cr}×)...")
        self.alloc_k = self.dp_alloc(self.sigma_k, CROSS_LAYER_DIM, "Keys")
        self.alloc_v = self.dp_alloc(self.sigma_v, CROSS_LAYER_DIM, "Values")


# ============================================================================
# Compressor
# ============================================================================
class KVTCCompressor:
    def __init__(self, cal, cfg, dev):
        self.cal, self.cfg, self.dev = cal, cfg, dev

    def _quant(self, v, nb):
        if nb == 0: return None, 0.0, 0.0
        nl = (1 << nb) - 1
        mn, mx = float(v.min()), float(v.max())
        sc = (mx - mn) / max(nl, 1)
        if sc == 0: sc = 1.0
        return torch.clamp(torch.round((v - mn) / sc), 0, nl).to(torch.int32), mn, sc

    def _dequant(self, q, mn, sc):
        return q.float() * sc + mn

    def _compress_mat(self, data, V, mu, alloc):
        nt = data.shape[0]
        D = (data.float() - mu.unsqueeze(0).to(data.device)) @ V.to(data.device)
        active = np.where(alloc > 0)[0]
        parts = []
        for idx in active:
            q, mn, sc = self._quant(D[:, idx].cpu(), int(alloc[idx]))
            parts.append(struct.pack("<HBff", idx, alloc[idx], mn, sc))
            parts.append(q.numpy().astype(np.uint16).tobytes())
        raw = b"".join(parts)
        if HAS_NVCOMP:
            rt = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
            comp = bytes(nvcomp.Codec(algorithm="Deflate").encode(nvcomp.as_array(rt)).cpu())
        else:
            comp = zlib.compress(raw, level=6)
        return {"compressed": comp, "raw_size": len(raw), "n_tokens": nt,
                "n_active": len(active), "gpu": HAS_NVCOMP}

    def _decompress_mat(self, c, V, mu, alloc):
        nt, rank = c["n_tokens"], V.shape[1]
        if HAS_NVCOMP and c.get("gpu"):
            ct = torch.frombuffer(bytearray(c["compressed"]), dtype=torch.uint8).cuda()
            raw = bytes(nvcomp.Codec(algorithm="Deflate").decode(nvcomp.as_array(ct)).cpu())
        else:
            raw = zlib.decompress(c["compressed"])
        D = torch.zeros(nt, rank, device=self.dev)
        active = np.where(alloc > 0)[0]
        off = 0
        for _ in active:
            idx, nb, mn, sc = struct.unpack("<HBff", raw[off:off+11]); off += 11
            q = np.frombuffer(raw[off:off+nt*2], dtype=np.uint16); off += nt*2
            D[:, idx] = self._dequant(torch.from_numpy(q.copy()).to(self.dev), mn, sc)
        return D @ V.T.to(self.dev) + mu.unsqueeze(0).to(self.dev)

    def compress(self, past):
        pk, pv = extract_kv(past)
        sl = pk[0].shape[2]
        sink, win = self.cfg.sink_tokens, min(self.cfg.window_tokens, sl - self.cfg.sink_tokens)
        cs, ce = sink, sl - win
        nc = ce - cs
        if nc <= 0:
            return None
        esz = pk[0].element_size()
        cos_c, sin_c = build_rope_cache(sl, HEAD_DIM, self.dev)

        t0 = time.time()
        kl, vl = [], []
        for li in range(N_LAYERS):
            kl.append(pk[li][0, :, cs:ce, :].permute(1, 0, 2))
            vl.append(pv[li][0, :, cs:ce, :].permute(1, 0, 2))
        cr, sr = cos_c[cs:ce], sin_c[cs:ce]
        for li in range(N_LAYERS):
            kl[li] = remove_rope_batch(kl[li], cr, sr)
        km = torch.cat([k.reshape(nc, -1) for k in kl], dim=1)
        vm = torch.cat([v.reshape(nc, -1) for v in vl], dim=1)

        ck = self._compress_mat(km, self.cal.V_k, self.cal.mu_k, self.cal.alloc_k)
        cv = self._compress_mat(vm, self.cal.V_v, self.cal.mu_v, self.cal.alloc_v)
        t_comp = time.time() - t0

        sc, wc = [], []
        sb = wb = 0
        for li in range(N_LAYERS):
            sk, sv = pk[li][:,:,:sink,:].clone(), pv[li][:,:,:sink,:].clone()
            wk, wv = pk[li][:,:,ce:,:].clone(), pv[li][:,:,ce:,:].clone()
            sc.append((sk, sv)); wc.append((wk, wv))
            sb += (sk.nelement()+sv.nelement())*esz
            wb += (wk.nelement()+wv.nelement())*esz

        comp_b = len(ck["compressed"]) + len(cv["compressed"])
        orig_b = sum((pk[l].nelement()+pv[l].nelement())*esz for l in range(N_LAYERS))

        return {"ck": ck, "cv": cv, "sink": sc, "win": wc,
                "cs": cs, "ce": ce, "sl": sl,
                "orig_bytes": orig_b,
                "comp_bytes": comp_b + sb + wb,
                "compress_ms": t_comp * 1000}

    def decompress(self, c):
        cs, ce, sl = c["cs"], c["ce"], c["sl"]
        nc = ce - cs
        cos_c, sin_c = build_rope_cache(sl, HEAD_DIM, self.dev)

        t0 = time.time()
        km = self._decompress_mat(c["ck"], self.cal.V_k, self.cal.mu_k, self.cal.alloc_k)
        vm = self._decompress_mat(c["cv"], self.cal.V_v, self.cal.mu_v, self.cal.alloc_v)

        rk, rv = [], []
        cr, sr = cos_c[cs:ce], sin_c[cs:ce]
        for li in range(N_LAYERS):
            sk, sv = c["sink"][li]; wk, wv = c["win"][li]
            fs = li * N_KV_HEADS * HEAD_DIM
            lk = km[:, fs:fs+N_KV_HEADS*HEAD_DIM].reshape(nc, N_KV_HEADS, HEAD_DIM)
            lv = vm[:, fs:fs+N_KV_HEADS*HEAD_DIM].reshape(nc, N_KV_HEADS, HEAD_DIM)
            lk = apply_rope_batch(lk, cr, sr)
            ck_ = lk.unsqueeze(0).permute(0,2,1,3).to(sk.dtype)
            cv_ = lv.unsqueeze(0).permute(0,2,1,3).to(sv.dtype)
            rk.append(torch.cat([sk, ck_, wk], dim=2))
            rv.append(torch.cat([sv, cv_, wv], dim=2))

        t_dec = time.time() - t0
        return rk, rv, t_dec * 1000


# ============================================================================
# RAG simulation
# ============================================================================
RAG_SYSTEM = """You are a helpful technical assistant. Answer questions based only on the provided context documents. Be concise and accurate.

## Document 1: KV Cache Management in LLM Inference
Key-value caches store intermediate computations in transformer models during autoregressive generation. For a model with L layers, H heads, and head dimension D, the KV cache for T tokens occupies 4*L*H*D*T bytes in 16-bit precision. Modern serving systems use paged attention to manage cache memory in blocks, similar to virtual memory in operating systems. Prefix sharing allows multiple requests with common prefixes to reuse cached computations, which is particularly valuable for shared system prompts and RAG contexts. When GPU memory is exhausted, caches must be evicted following an LRU policy, forcing expensive recomputation on the next request. The tension between retaining caches for future reuse and freeing memory for new requests creates a fundamental latency-throughput dilemma in production systems. Block-level paging divides the KV cache into fixed-size blocks that can be allocated and freed independently, enabling fine-grained memory management without the fragmentation issues of contiguous allocation. Each block typically holds 16-128 tokens worth of key-value pairs. The block table maps logical token positions to physical block locations, allowing non-contiguous storage while maintaining the appearance of a contiguous cache to the attention mechanism. This approach was pioneered by vLLM's PagedAttention and has since been adopted by most production serving frameworks.

## Document 2: Compression Techniques for KV Caches
FP8 quantization reduces KV cache from 16-bit to 8-bit, achieving a fixed 2x compression with minimal accuracy loss. This is the simplest approach and is natively supported by modern hardware like NVIDIA H100 and AMD MI300X GPUs. The E4M3 format with 4 exponent bits and 3 mantissa bits provides sufficient precision for most inference workloads. Token eviction methods like H2O (Heavy-Hitter Oracle) and TOVA selectively discard less important tokens based on attention scores accumulated during generation. H2O maintains a fixed budget of recent tokens plus "heavy hitter" tokens that receive disproportionate attention weight. While effective at moderate compression ratios of 4-8x, these methods can catastrophically fail on tasks requiring precise retrieval from specific positions in the context. SVD-based methods like xKV exploit the low-rank structure of KV caches by computing singular value decomposition and retaining only the top-k singular vectors. However, these methods require computing a separate SVD for each prompt, which adds significant latency during prefill. The computational cost of SVD scales cubically with the matrix dimensions, making it impractical for very long contexts. Transform coding, as implemented in kvtc, takes a fundamentally different approach by computing a single PCA basis during offline calibration that generalizes across all prompts. This amortizes the expensive decomposition cost and enables much faster compression at inference time. The key insight is that KV caches across different attention layers and heads exhibit strong correlations that can be exploited through cross-layer concatenation before applying PCA. The resulting principal components are then quantized with variable bit widths determined by a dynamic programming algorithm that minimizes reconstruction error under a fixed bit budget.

## Document 3: Production Serving Architecture for LLMs
Large-scale LLM serving architectures often disaggregate prefill and decode across separate node pools connected by high-speed RDMA-capable fabric such as InfiniBand or RoCE. This separation is motivated by the fundamentally different compute profiles of the two phases: prefill is compute-bound with high arithmetic intensity, while decode is memory-bandwidth-bound with low arithmetic intensity. Prefill nodes produce KV caches and transmit them to decode nodes, where the actual token-by-token generation occurs. Both node types maintain tiered KV cache hierarchies: GPU HBM serves as the hot tier for actively used caches, CPU DRAM provides a warm tier for recently used caches that may be needed again soon, and NVMe SSDs or network-attached storage serve as the cold tier for long-term cache retention. A cache-aware router sits between the client and the node pools, directing incoming requests to nodes that already hold matching prefix caches whenever possible. This routing optimization can eliminate prefill entirely for repeated system prompts, reducing time-to-first-token from seconds to milliseconds. The dominant cross-node traffic in disaggregated serving is KV cache transfer, making compression critical for network efficiency. For a 70B model processing 8K tokens, the KV cache is approximately 2.5 GiB, which at 400 Gbps InfiniBand bandwidth takes about 50ms to transfer. With 20x compression, this drops to 2.5ms, essentially eliminating the transfer bottleneck. Cache lifetime management follows a priority-based eviction policy where caches are scored based on recency, frequency of access, and remaining TTL. Prefix trees enable efficient lookup of shared prefixes across cached entries. The system must balance the cost of cache retention (memory/storage) against the cost of recomputation (compute + latency). This tradeoff becomes increasingly favorable for compression as model sizes grow, since recomputation cost scales with model parameters while compression overhead scales only with cache size.

## Document 4: Benchmarking Results and Performance Analysis
Comprehensive benchmarking of KV cache compression methods reveals significant performance differences across compression ratios and task types. On Llama 3.1 8B with standard benchmarks, transform coding (kvtc) at 16x target compression achieves 18-22x actual compression after entropy coding, while maintaining GSM8K accuracy within 0.1 points of the uncompressed baseline (56.9 vs 56.8). On MMLU, the accuracy difference is similarly negligible (60.1 vs 60.5). Long-context benchmarks show the strongest differentiation between methods. On the Lost in the Middle key-value retrieval task with 100 keys, kvtc at 16x maintains 99.3% accuracy compared to the 99.4% baseline, while KIVI 2-bit drops to 88.8% and H2O collapses to 20.2%. This demonstrates that transform coding preserves the fine-grained positional information needed for precise retrieval, unlike token eviction methods that may discard the very tokens containing the answer. For reasoning models like DeepSeek-R1-distilled Qwen 2.5 7B, kvtc at 8x compression shows AIME 2024 scores of 52.5 compared to the 50.9 baseline, actually slightly improving due to regularization effects. On LiveCodeBench coding tasks, the accuracy drop is minimal at 0.2 percentage points. Latency measurements on Mistral NeMo 12B running on an H100 GPU show that kvtc decompression takes 267ms for 8K context at batch size 8, compared to 3098ms for full recomputation — an 8x improvement in time-to-first-token. The compression pipeline itself takes 379ms, meaning the round-trip compress-store-decompress cycle is still 5.6x faster than recomputation. For batch size 2 with 16K context, decompression takes 143ms versus 1780ms for recomputation, a 12.4x improvement.

## Document 5: Multi-Turn Conversation and Cache Reuse Patterns
In production chat deployments, the majority of compute is spent on processing shared context rather than unique user queries. Analysis of conversation logs from coding assistants shows that the average session involves 8-12 turns, with the system prompt and file context comprising 70-90% of the total token count. Each turn appends only 50-200 new tokens (the user query) to the existing context. Without cache reuse, every turn requires full recomputation of the entire context, leading to O(T^2) total compute across the session. With prefix caching, only the new tokens need processing, reducing per-turn cost to O(T * delta_T). However, prefix caching requires the KV cache to remain in GPU memory between turns. In a multi-tenant serving environment with hundreds of concurrent sessions, the aggregate KV cache memory requirements quickly exceed available HBM capacity. This forces either frequent cache eviction (increasing recomputation) or limiting the number of concurrent sessions (reducing throughput). KV cache compression addresses this by reducing the memory footprint of retained caches, allowing more sessions to maintain their cached context. At 20x compression, a system that could previously hold 10 active sessions can now effectively hold 200 sessions worth of compressed caches, dramatically improving the cache hit rate and reducing average latency across all users.
"""

RAG_QUESTIONS = [
    "What is the formula for KV cache size in bytes?",
    "How does transform coding compare to FP8 quantization in compression ratio?",
    "What is the role of cache-aware routing in production serving?",
    "Why is decompression faster than recomputation for time-to-first-token?",
    "What compression ratio does KIVI achieve and what are its limitations?",
]


def gpu_mem_used():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def fmt(b):
    if b < 1024: return f"{b} B"
    if b < 1048576: return f"{b/1024:.1f} KiB"
    return f"{b/1048576:.2f} MiB"


def generate_response(model, tokenizer, input_ids, past_kv=None, max_new=100):
    """Generate with optional past KV cache."""
    with torch.no_grad():
        out = model.generate(
            input_ids,
            past_key_values=past_kv,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================================
# Main RAG simulation
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--target-cr", type=int, default=16)
    ap.add_argument("--num-turns", type=int, default=4)
    ap.add_argument("--calibration-samples", type=int, default=32)
    ap.add_argument("--max-cal-len", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = torch.device("cuda" if args.device == "auto" and
                       torch.cuda.is_available() else "cpu")
    entropy = "nvCOMP GPU" if HAS_NVCOMP else "zlib CPU"
    print(f"{'='*70}")
    print(f"  kvtc RAG PoC — Multi-turn with KV Cache Compression")
    print(f"{'='*70}")
    print(f"  Device: {dev}  |  DEFLATE: {entropy}")

    # Load model
    print(f"\n[Setup] Loading {args.model_id}...")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=dev)
    model.eval()

    # Calibrate kvtc
    print(f"\n[Setup] Calibrating kvtc (CR={args.target_cr}×)...")
    cfg = KVTCConfig(target_cr=args.target_cr)
    cal = KVTCCalibrator(cfg, dev)
    # Use the RAG system prompt + variations as calibration data
    cal_texts = [RAG_SYSTEM] * 8 + [
        RAG_SYSTEM + f"\nQuestion: {q}\nAnswer:" for q in RAG_QUESTIONS
    ] * 4
    cal_texts = cal_texts[:max(args.calibration_samples, 24)]
    cal.calibrate(model, tok, cal_texts, max_len=args.max_cal_len)
    compressor = KVTCCompressor(cal, cfg, dev)

    # Build the RAG context
    context = RAG_SYSTEM
    questions = RAG_QUESTIONS[:args.num_turns]

    # Tokenize context once
    ctx_ids = tok(context, return_tensors="pt").to(dev)
    ctx_len = ctx_ids["input_ids"].shape[1]
    print(f"\n[RAG Context] {ctx_len} tokens")

    # =======================================================================
    # Strategy A: RECOMPUTE — discard cache every turn
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  Strategy A: RECOMPUTE (discard cache between turns)")
    print(f"{'='*70}")
    recompute_times = []
    for i, q in enumerate(questions):
        full_prompt = context + f"\n\nQuestion: {q}\nAnswer:"
        ids = tok(full_prompt, return_tensors="pt", truncation=True,
                  max_length=4096).to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model(**ids, use_cache=True)
        # Simulate generation of first token (TTFT = prefill time)
        ttft = (time.time() - t0) * 1000
        recompute_times.append(ttft)
        total_tok = ids["input_ids"].shape[1]
        mem = kv_memory_bytes(out.past_key_values)
        print(f"  Turn {i+1}: \"{q[:50]}...\"")
        print(f"    Prefill {total_tok} tokens → TTFT: {ttft:.0f}ms, KV: {fmt(mem)}")
        del out; torch.cuda.empty_cache()

    # =======================================================================
    # Strategy B: HOLD — keep cache in HBM
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  Strategy B: HOLD (keep cache in HBM between turns)")
    print(f"{'='*70}")

    # Prefill context once
    t0 = time.time()
    with torch.no_grad():
        ctx_out = model(**ctx_ids, use_cache=True)
    ctx_prefill_ms = (time.time() - t0) * 1000
    ctx_past = ctx_out.past_key_values
    ctx_mem = kv_memory_bytes(ctx_past)
    print(f"  Context prefill: {ctx_len} tokens, {ctx_prefill_ms:.0f}ms, KV: {fmt(ctx_mem)}")

    hold_times = []
    hold_mem_between = []
    for i, q in enumerate(questions):
        q_text = f"\n\nQuestion: {q}\nAnswer:"
        q_ids = tok(q_text, return_tensors="pt").to(dev)

        mem_before = gpu_mem_used()
        hold_mem_between.append(mem_before)

        t0 = time.time()
        # Only prefill the question (cache has context)
        with torch.no_grad():
            out = model(q_ids["input_ids"], past_key_values=ctx_past, use_cache=True)
        ttft = (time.time() - t0) * 1000
        hold_times.append(ttft)
        q_tok = q_ids["input_ids"].shape[1]
        print(f"  Turn {i+1}: \"{q[:50]}...\"")
        print(f"    Prefill {q_tok} new tokens → TTFT: {ttft:.0f}ms, "
              f"GPU mem: {mem_before:.0f} MiB (cache held)")
        del out

    del ctx_past, ctx_out; torch.cuda.empty_cache(); gc.collect()

    # =======================================================================
    # Strategy C: kvtc — compress between turns
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  Strategy C: kvtc (compress cache between turns)")
    print(f"{'='*70}")

    # Prefill context
    t0 = time.time()
    with torch.no_grad():
        ctx_out = model(**ctx_ids, use_cache=True)
    ctx_prefill_ms2 = (time.time() - t0) * 1000
    ctx_past = ctx_out.past_key_values
    ctx_mem = kv_memory_bytes(ctx_past)
    print(f"  Context prefill: {ctx_len} tokens, {ctx_prefill_ms2:.0f}ms, KV: {fmt(ctx_mem)}")

    # Compress context cache
    compressed = compressor.compress(ctx_past)
    if compressed is None:
        print("  ERROR: sequence too short for compression")
        return

    print(f"  Compressed: {fmt(compressed['orig_bytes'])} → {fmt(compressed['comp_bytes'])} "
          f"({compressed['orig_bytes']/compressed['comp_bytes']:.1f}×) "
          f"in {compressed['compress_ms']:.0f}ms")

    # Free original cache
    del ctx_past, ctx_out
    torch.cuda.empty_cache(); gc.collect()
    mem_after_free = gpu_mem_used()
    print(f"  GPU memory after freeing cache: {mem_after_free:.0f} MiB")

    kvtc_times = []
    kvtc_decomp_times = []
    for i, q in enumerate(questions):
        q_text = f"\n\nQuestion: {q}\nAnswer:"
        q_ids = tok(q_text, return_tensors="pt").to(dev)

        # Decompress
        t0 = time.time()
        rk, rv, dec_ms = compressor.decompress(compressed)
        # Build a DynamicCache-compatible structure
        from transformers.cache_utils import DynamicCache
        restored_cache = DynamicCache()
        for li in range(N_LAYERS):
            # update() expects [batch, heads, new_seq, dim] — 4D
            # rk[li] is already [1, heads, seq, dim]
            restored_cache.update(rk[li], rv[li], li)

        # Prefill question only
        with torch.no_grad():
            out = model(q_ids["input_ids"], past_key_values=restored_cache,
                        use_cache=True)
        total_ms = (time.time() - t0) * 1000
        ttft_effective = total_ms  # decompress + question prefill

        kvtc_times.append(ttft_effective)
        kvtc_decomp_times.append(dec_ms)
        q_tok = q_ids["input_ids"].shape[1]
        print(f"  Turn {i+1}: \"{q[:50]}...\"")
        print(f"    Decompress: {dec_ms:.0f}ms + prefill {q_tok} tokens "
              f"→ TTFT: {ttft_effective:.0f}ms")
        del out, restored_cache, rk, rv
        torch.cuda.empty_cache()

    # =======================================================================
    # Summary
    # =======================================================================
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  Context: {ctx_len} tokens | Turns: {len(questions)} | CR: {args.target_cr}×")
    print(f"")
    print(f"  {'':>30} {'Recompute':>12} {'Hold':>12} {'kvtc':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    avg_r = sum(recompute_times) / len(recompute_times)
    avg_h = sum(hold_times) / len(hold_times)
    avg_k = sum(kvtc_times) / len(kvtc_times)
    print(f"  {'Avg TTFT (ms)':>30} {avg_r:>10.0f}ms {avg_h:>10.0f}ms {avg_k:>10.0f}ms")

    print(f"  {'KV cache in HBM':>30} {'0 (freed)':>12} {fmt(ctx_mem):>12} {'0 (freed)':>12}")
    print(f"  {'Compressed storage':>30} {'N/A':>12} {'N/A':>12} {fmt(compressed['comp_bytes']):>12}")
    print(f"  {'Recompute savings':>30} {'baseline':>12} "
          f"{avg_r/max(avg_h,0.01):.1f}× faster:>0 "
          f"{avg_r/max(avg_k,0.01):.1f}× faster")
    print(f"")

    if avg_k < avg_r:
        speedup = avg_r / avg_k
        print(f"  ✓ kvtc is {speedup:.1f}× faster than recomputing")
    else:
        slowdown = avg_k / avg_r
        print(f"  ✗ kvtc is {slowdown:.1f}× slower than recomputing "
              f"(context too short — need longer sequences)")

    print(f"  ✓ kvtc frees {fmt(compressed['orig_bytes'])} of HBM between turns")
    print(f"  ✓ Compressed cache: {fmt(compressed['comp_bytes'])} "
          f"(vs {fmt(compressed['orig_bytes'])} original)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
