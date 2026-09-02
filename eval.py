import os
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B"

ALPACA_INFER = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

hf_token = os.environ.get("HF_TOKEN")


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=hf_token, dtype=torch.bfloat16, device_map="auto"
)

adapter = snapshot_download(
    repo_id="tardelr/lora-dora-reproducibility",
    allow_patterns="dora/seed-42/final_adapter/*",
)
model = PeftModel.from_pretrained(model, f"{adapter}/dora/seed-42/final_adapter")
model.eval()

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["arc_challenge", "hellaswag", "winogrande", "boolq", "piqa", "openbookqa"],
    num_fewshot=0,
    log_samples=True,
)
print(lm_eval.utils.make_table(results))

# basic prompt testing
# prompt = ALPACA_INFER.format(instruction="Will it rain today?")
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     out = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         do_sample=False,          # greedy — deterministic, what you want for a repro study
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

# completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
# print(completion)