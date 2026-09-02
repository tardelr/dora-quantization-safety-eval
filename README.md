# DoRA / LoRA Quantization Safety Eval

**Status: WIP ⚠️**

Does [DoRA](https://arxiv.org/abs/2402.09353) preserve safety behavior better than LoRA (or other PEFT adapters), and does quantization change that picture?

## Questions

1. **Adapter effect** — after identical instruction tuning, do DoRA and LoRA adapters differ in safety behavior (refusal rates, harmful completions, jailbreak resistance)?
2. **Quantization effect** — does serving the same adapter at reduced precision (e.g. 4-bit vs bf16) degrade safety alignment, and is one adapter type more robust to it?

## Setup

- **Base model:** `meta-llama/Meta-Llama-3-8B`
- **Adapters:** [`tardelr/lora-dora-reproducibility`](https://huggingface.co/tardelr/lora-dora-reproducibility) — multiple adapter types × seeds
- **Training pipeline:** [tardelr/dora-case-studies](https://github.com/tardelr/dora-case-studies) (adapters here are trained artifacts of that pipeline, not retrained)

## Usage

```bash
export HF_TOKEN=...   # needed for gated Llama 3 weights
python eval.py
```

`eval.py` currently loads one adapter (`dora/seed-42`) and runs the capability benchmark sweep. Safety evals and the quantization axis are not wired up yet.
