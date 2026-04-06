"""
evaluate_model.py
──────────────────
Evaluate the fine-tuned Légifrance model using ROUGE-L, BERTScore,
and article citation accuracy.

Usage:
    python scripts/evaluate_model.py
"""

import re
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER  = Path("outputs/lora-legifrance/final")
EVAL_N   = 50

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


def generate(instruction: str) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def parse_sample(text: str):
    parts = text.split("### Response:\n", 1)
    if len(parts) == 2:
        instr = parts[0].replace("### Instruction:\n", "").strip()
        return instr, parts[1].strip()
    return "", text.strip()


def cited_articles(text: str) -> set[str]:
    pattern = re.compile(r"Article\s+([LRD]?\d[\d\-]*)", re.IGNORECASE)
    return set(pattern.findall(text))

eval_path = Path("data/eval.jsonl")
samples   = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
samples   = samples[:EVAL_N]

instructions, references = zip(*[parse_sample(s["text"]) for s in samples])

print(f"Evaluating on {len(samples)} samples...")
predictions = [generate(instr) for instr in instructions]

rouge     = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

rouge_result = rouge.compute(
    predictions=list(predictions),
    references=list(references),
)
bs_result = bertscore.compute(
    predictions=list(predictions),
    references=list(references),
    lang="fr",
    model_type="camembert-base",
    num_layers=9,
    device=DEVICE
)
citation_hits = [
    bool(cited_articles(p) & cited_articles(r))
    for p, r in zip(predictions, references)
]
citation_acc = sum(citation_hits) / len(citation_hits)

print("\n── Evaluation Results ──────────────────────────────")
print(f"  ROUGE-1         : {rouge_result['rouge1']:.4f}")
print(f"  ROUGE-2         : {rouge_result['rouge2']:.4f}")
print(f"  ROUGE-L         : {rouge_result['rougeL']:.4f}")
print(f"  BERTScore F1 fr : {sum(bs_result['f1'])/len(bs_result['f1']):.4f}")
print(f"  Citation accuracy: {citation_acc:.2%}")
