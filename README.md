# kvtc PoC — KV Cache Compression for LLM Inference

A proof-of-concept implementation of NVIDIA's **kvtc** (KV Cache Transform Coding) paper ([arXiv:2511.01815](https://arxiv.org/abs/2511.01815)) applied to **Llama 3.2 1B**.

## What problem does this solve?

When an LLM generates text, it stores intermediate computations called the **KV cache** so it doesn't have to redo work for every new token. This cache grows linearly with conversation length and eats GPU memory fast — memory that could be serving other users.

kvtc **compresses** this cache using techniques borrowed from image/video compression (think JPEG for LLM memory), achieving **~5-20× size reduction** while keeping the model's output quality nearly identical.

## How it works

```
KV Cache → Remove positional info → PCA projection → Quantize → DEFLATE → Storage
                                                                              │
Storage → Inflate → Dequantize → Inverse PCA → Re-apply positions → KV Cache ◄┘
```

1. **Calibration** (once per model): Run sample texts through the model, collect KV caches, and learn a compression basis via PCA
2. **Compress**: Project the cache into a compact space, quantize with variable precision, and apply lossless compression
3. **Decompress**: Reverse the process to reconstruct the cache when needed

The key insight is that KV caches across different layers and attention heads are highly correlated — PCA exploits this redundancy.

## Results (Llama 3.2 1B, 862 tokens, CR=16×)

| Metric | Value |
|---|---|
| KV cache before | 26.94 MiB |
| KV cache after | 5.78 MiB |
| Space saved | 78.5% |
| Key reconstruction (cosine sim) | 0.9904 |
| Value reconstruction (cosine sim) | 0.8636 |
| Compression time | 835 ms |
| Decompression time | 347 ms |

The overall compression ratio improves with longer sequences since the fixed-size uncompressed window (128 tokens) becomes a smaller fraction of the total.

## Quick start

```bash
pip install torch transformers accelerate datasets scikit-learn numpy huggingface-hub

python kvtc_poc.py --calibration-samples 128 --target-cr 16 --max-cal-len 2048
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--calibration-samples` | 128 | Number of texts for calibration |
| `--target-cr` | 16 | Target compression ratio |
| `--max-cal-len` | 2048 | Max tokens per calibration sample |
| `--pca-rank` | 4096 | PCA dimensionality |
| `--model-id` | `meta-llama/Llama-3.2-1B` | HuggingFace model ID |

## Limitations

This is a PoC, not production code:

- **CPU DEFLATE** via Python's zlib — production would use NVIDIA nvCOMP on GPU
- **Greedy DP** for bit allocation — the paper uses a full dynamic programming algorithm with joint group-size optimization
- **Synthetic calibration data** — the paper uses 160K tokens from FineWeb + OpenR1Math
- **No vLLM integration** — would need a KV Connector or LMCache backend for serving

## References

- [KV Cache Transform Coding for Compact Storage in LLM Inference](https://arxiv.org/abs/2511.01815) — Staniszewski & Łańcucki, NVIDIA (ICLR 2026)
- [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) — Related KV cache compression library
- [LMCache](https://github.com/LMCache/LMCache) — KV cache management for vLLM
