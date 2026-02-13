# kvtc-test — KV Cache Compression for LLM Inference

Proof-of-concept implementation of NVIDIA's **kvtc** (KV Cache Transform Coding) paper ([arXiv:2511.01815](https://arxiv.org/abs/2511.01815)), applied to **Llama 3.2 1B**.

## What problem does this solve?

When an LLM generates text, it stores intermediate computations called the **KV cache** to avoid redoing work for every new token. This cache grows with conversation length and eats GPU memory — memory that could be serving other users.

kvtc **compresses** this cache using techniques from image/video compression (think JPEG for LLM memory), achieving **~5-20× size reduction** while keeping output quality nearly identical.

This is **not a replacement** for vLLM's FP8 KV cache quantization (which gives 2× at zero cost during inference). kvtc is designed for **storage and offload** — compressing caches between conversation turns, across nodes, or to CPU/SSD so GPU memory is freed for other requests.

## How it works

```
KV Cache → Remove RoPE → PCA projection → Quantize → DEFLATE → Compressed storage
                                                                        │
Compressed storage → Inflate → Dequantize → Inverse PCA → Re-apply RoPE → KV Cache
```

1. **Calibration** (once per model) — run sample texts, collect KV caches, learn a PCA basis
2. **Compress** — project into compact space, quantize with variable precision, entropy code
3. **Decompress** — reverse the pipeline to restore the cache

## Repository structure

| File | Description |
|---|---|
| `kvtc_poc.py` | Core implementation — calibration, compression, reconstruction quality report (CPU zlib DEFLATE) |
| `kvtc_poc_gpu.py` | Same as above with **nvCOMP GPU DEFLATE** when `nvidia-nvcomp-cu12` is installed |
| `kvtc_rag_poc.py` | RAG multi-turn simulation comparing 3 strategies: recompute vs hold-in-HBM vs kvtc compress/decompress |

## Quick start

```bash
pip install torch transformers accelerate datasets scikit-learn numpy huggingface-hub

# Basic compression test
python kvtc_poc.py --calibration-samples 128 --target-cr 16 --max-cal-len 2048

# With GPU DEFLATE (optional)
pip install nvidia-nvcomp-cu12
python kvtc_poc_gpu.py --calibration-samples 128 --target-cr 16 --max-cal-len 2048

# RAG multi-turn simulation
python kvtc_rag_poc.py --target-cr 16 --num-turns 4
```

## Results (Llama 3.2 1B-Instruct, 862 tokens, CR=16×)

| Metric | CPU (zlib) | GPU (nvCOMP) |
|---|---|---|
| KV cache before | 26.94 MiB | 26.94 MiB |
| KV cache after | 5.78 MiB | 6.01 MiB |
| Space saved | 78.5% | 77.7% |
| Key cosine similarity | 0.9904 | 0.9904 |
| Value cosine similarity | 0.8636 | 0.8636 |
| Compress time | 835 ms | 779 ms |
| Decompress time | 347 ms | 496 ms |
| DEFLATE backend | zlib CPU | nvCOMP GPU |

Overall compression ratio improves with longer sequences since the fixed-size uncompressed window (128 tokens) becomes a smaller fraction.

## Where kvtc makes sense

kvtc shines in scenarios where KV caches need to be **stored, moved, or retained**:

- **Multi-turn chat** — compress the cache while the user is typing, decompress when they send
- **RAG with shared context** — compress the document context once, decompress per question
- **Disaggregated serving** — transfer compressed caches between prefill and decode nodes
- **Cache tiering** — keep compressed caches in CPU RAM/SSD instead of evicting from HBM

For short contexts (<500 tokens) on fast GPUs, plain recomputation is faster than decompress. The crossover happens around **2K-4K+ tokens** for 8B models and earlier for larger models where prefill cost is higher.

## Limitations

This is a PoC, not production code:

- **Greedy DP** for bit allocation — the paper uses a full dynamic programming algorithm
- **Synthetic calibration data** — production would use 160K tokens from diverse corpora
- **No vLLM integration** — would need a KV Connector or LMCache backend
- **Single-sequence** — no batched compression/decompression

## Options

| Flag | Default | Description |
|---|---|---|
| `--calibration-samples` | 128 (poc) / 32 (rag) | Calibration texts |
| `--target-cr` | 16 | Target compression ratio |
| `--max-cal-len` | 2048 | Max tokens per calibration sample |
| `--pca-rank` | 4096 | PCA dimensionality |
| `--num-turns` | 4 | RAG questions (rag only) |

## References

- [KV Cache Transform Coding for Compact Storage in LLM Inference](https://arxiv.org/abs/2511.01815) — Staniszewski & Łańcucki, NVIDIA (ICLR 2026)
- [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) — KV cache compression library
- [LMCache](https://github.com/LMCache/LMCache) — KV cache layer for vLLM
- [nvCOMP](https://developer.nvidia.com/nvcomp) — GPU-accelerated compression

## License

See [LICENSE](LICENSE).
