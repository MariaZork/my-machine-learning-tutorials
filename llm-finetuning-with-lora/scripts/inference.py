"""
inference.py
────────────
Load a trained LoRA adapter and run inference on the Légifrance model.

Usage:
    python scripts/inference.py
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID   = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER    = Path("outputs/lora-legifrance/final")

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base  = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
model = PeftModel.from_pretrained(base, str(ADAPTER)).to(DEVICE).eval()

# ── Option: merge adapter (zero inference overhead) ───────────────────────────
# merged = model.merge_and_unload()
# merged.save_pretrained("outputs/lora-legifrance-merged")


def generate(instruction: str, max_new_tokens: int = 300) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens   = max_new_tokens,
            temperature      = 0.1,
            do_sample        = True,
            repetition_penalty= 1.1,
            pad_token_id     = tokenizer.eos_token_id,
        )
    new_toks = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True)


QUERIES = [
    "What are the employer obligations under Article L4121-1 of the Code du travail?",
    "Summarise the provisions on rupture conventionnelle (Article L1237-19).",
    "What rights does Article L3121-27 grant workers regarding working hours?",
    "Explain Article L1242-1 on fixed-term employment contracts (CDD).",
]

if __name__ == "__main__":
    for q in QUERIES:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print(f"A: {generate(q)}")
