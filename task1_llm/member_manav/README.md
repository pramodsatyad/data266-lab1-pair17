# Task 1: Character-Level GPT on TinyStories

Task 1 asks us to build a small GPT-style language model from scratch and train it to write simple children's stories. I wrote the model by hand at the character level: no library attention, no pretrained weights. It reads TinyStories one character at a time and learns to predict the next character. This file explains what the model is, how it works, what I tried, and how to run it.

## Result

Measured on the full GPU run (100,000 train stories, 10,000 val stories, 10 epochs):

| Metric | Value |
|---|---|
| Final train loss | 0.5388 |
| Final val loss | 0.5332 |
| Perplexity (val) | 1.7044 |
| Top-1 accuracy (val) | 0.8289 |
| Generalization gap | -0.0056 |

A small negative generalization gap means val loss is slightly below train loss, which is normal here (train loss is averaged over the whole epoch including early, worse steps; val is measured once at the end with the better weights and no dropout). Generated text quality: greedy decoding can get stuck repeating the same sentence, and the model can lose track of who a character is partway through a story (see `failure_analysis.md`).

## How it works

```mermaid
flowchart LR
    T[story text] --> C[character tokens]
    C --> E[token + position embedding]
    E --> B1[transformer block 1]
    B1 --> B2[...]
    B2 --> B6[transformer block 6]
    B6 --> LN[final layernorm]
    LN --> H[LM head: linear to vocab]
    H --> N[predicted next character]
```

- **The model is a GPT written by hand** (`src/model.py`): 6 transformer blocks, each with pre-LayerNorm, causal self-attention, a residual connection, pre-LayerNorm, a feed-forward layer, and another residual connection. No `nn.MultiheadAttention`, no `nn.Transformer`, no `scaled_dot_product_attention` — the attention and its causal mask are built from plain `nn.Linear` layers and matrix multiplies.
- **Tokens are single characters**, not words or subwords. Vocab size is 115 characters, built only from the train text.
- **Attention has 6 heads over a 384-dim embedding** (`n_head: 6`, `n_embd: 384` in `configs/gpu.yaml`), and each training example is 256 characters long (`block_size: 256`).
- **Training uses warmup then cosine decay** (1,000 warmup steps, then the learning rate decays to 10% of its peak of 6e-4), gradient clipping at 1.0, and bf16 autocast for roughly 2x speed on the RTX 4090 with almost no accuracy loss.
- **Total size: 10,827,379 parameters** (about 10.8 million), trained for 10 epochs.

## What I did, step by step

| Step | What changed | Result |
|---|---|---|
| Folder and configs | set up `local` (tiny, fast) and `gpu` (full) configs | — |
| Char tokenizer and data split | built the 115-character vocab and a fixed 100K/10K train/val split, seed 42 | — |
| GPT model from scratch | wrote attention, blocks, and the model by hand | — |
| Training loop | added LR warmup, cosine decay, checkpoints | — |
| Generation and metrics | added sampling (greedy and temperature) and the full metrics report | — |
| `torch.load` fix | newer torch versions needed `weights_only=False` to load checkpoints with optimizer/scheduler state | checkpoint loading works again |
| GPU run | full 10-epoch run on the RTX 4090 | val loss 0.5332, the result above |

## Problems I found and fixed

- **`torch.load` broke on a newer torch version.** `load_checkpoint` in `src/utils.py` called `torch.load(path, map_location=device)`, which newer torch versions reject by default for checkpoints that include optimizer/scheduler state. Fixed by adding `weights_only=False` to the call.
- **Greedy decoding gets stuck in repetition loops** (see `failure_analysis.md`, Failure 1). Not fixed yet — listed as future work (add top-k/top-p sampling or a repetition penalty).
- **The model mixes up who a character is partway through longer generated text** (see `failure_analysis.md`, Failure 2), because it only ever sees the last 256 characters. Not fixed yet — would need a longer block size.

## Folder map

| Path | What it is |
|---|---|
| `configs/local.yaml`, `configs/gpu.yaml` | smoke-test and full-run settings |
| `src/model.py` | the GPT model, written from scratch |
| `src/data.py` | TinyStories loading, character tokenizer, train/val split |
| `src/utils.py` | checkpoint save/load, logger, device helpers |
| `task1_gpt.ipynb` | the one notebook that trains, generates, and evaluates |
| `../eval_task1.py` | metrics and generation-quality helper functions used by the notebook |
| `checkpoints/manav_task1_weights.pt` | the trained model weights |
| `outputs/gpu/loss_curve.png`, `samples.txt`, `metrics_report.csv`, `train_log.txt` | the full run's plots, generated text, and metrics |
| `results.md` | the full write-up: architecture, hyperparameters, metrics |
| `failure_analysis.md` | 3 real generation failures, with causes |

## How to run

**(a) Quick smoke test** (tiny model, 2,000 train stories, 1 epoch):
```powershell
cd task1_llm/member_manav
$env:CONFIG = "local"
..\..\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task1_gpt.ipynb
```

**(b) Full training:**
```powershell
cd task1_llm/member_manav
$env:CONFIG = "gpu"
..\..\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task1_gpt.ipynb
```

**(c) Evaluation:** there is no separate evaluation script — `task1_gpt.ipynb` computes the loss, perplexity, accuracy, and generation metrics as part of the same run shown above.

## Rules I followed

- The GPT model is written entirely from scratch, with no library attention module (`nn.MultiheadAttention`, `nn.Transformer`, or `scaled_dot_product_attention`) anywhere.
- No pretrained weights are used anywhere in the model.
- The train/val split uses a fixed seed (42) so it is always the same.
