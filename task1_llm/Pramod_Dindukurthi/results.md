# Task 1 — GPT-Style LLM From Scratch

## Architecture

- Character-level autoregressive GPT
- Context length: 128
- Embedding dimension: 256
- Attention heads: 8
- Transformer blocks: 4
- Feed-forward dimension: 1024
- Dropout: 0.1
- Parameter count: 3,217,152

Attention was implemented manually using Q/K/V projections,
scaled dot-product matrix multiplication, and a causal lower-triangular mask.
No prebuilt Transformer or attention module was used.

## Data

- Dataset: TinyStories
- Training stories: 100,000
- Validation stories: 10,000
- Tokenization: character-level
- Vocabulary size: 113
- Sequence length: 128

## Training

- Epochs: 10
- Batch size: 128
- Gradient accumulation: 1
- Optimizer: AdamW
- Peak learning rate: 0.0004
- LR schedule: linear warm-up + cosine decay
- AMP: True
- Best validation epoch: 10

## Hardware

- GPU: NVIDIA GeForce RTX 4090
- PyTorch: 2.1.2
- CUDA runtime: 12.1

## Final Metrics

| Metric | Value |
|---|---:|
| Train cross-entropy | 0.6758 |
| Validation cross-entropy | 0.6852 |
| Perplexity | 1.984 |
| Bits per character | 0.989 |
| Generalization gap | 0.0094 |
| Top-1 next-character accuracy | 0.7813 |
| Distinct-1 | 0.0680 |
| Distinct-2 | 0.3202 |
| Distinct-3 | 0.5317 |
| Repeated 4-gram rate | 0.3596 |
| Mean gradient norm (best epoch) | 0.4087 |
| Max gradient norm (best epoch) | 0.4550 |
| Non-finite gradient events | 20 |
| Training tokens/sec (best epoch) | 988112.5 |
| Generation tokens/sec | 424.1 |
| Peak GPU memory (MB) | 1335.0 |
| Total training time (min) | 16.0 |

## Generated Samples

See:
- `outputs/generated_samples.txt`
- `outputs/generated_samples.csv`

## Failure Analysis

See `failure_analysis.md`.

## Discussion

The model converged smoothly across all 10 epochs, with validation loss continuing
to improve through the final epoch. The final train and validation losses remained
close, indicating a small generalization gap and no obvious overfitting.

Greedy decoding produced more repetitive outputs and frequently entered repeated
phrase loops. Temperature sampling at 0.8 increased sequence diversity, producing
higher Distinct-1/2/3 scores and substantially lower repeated 4-gram rates, although
it occasionally introduced character drift and semantic inconsistencies.

Training remained stable overall. No non-finite loss events occurred. A small number
of non-finite gradient-norm events were detected during mixed-precision training and
handled by AMP GradScaler without causing training divergence.
