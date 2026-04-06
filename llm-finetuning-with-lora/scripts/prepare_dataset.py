import re
import json
import random
from pathlib import Path
import requests
from datasets import Dataset

LEGI_RAW_BASE = (
    "https://raw.githubusercontent.com/SocialGouv/legi-data/master/data/"
)

CODES = {
    "Code du travail": "LEGITEXT000006072050",
    "Code de la sécurité sociale": "LEGITEXT000006073189",
    "Code rural et de la pêche maritime": "LEGITEXT000022197698",
    "Code des relations entre le public et l'administration": "LEGITEXT000031366350",
}

INSTRUCTION_TEMPLATES = [
    "Summarise Article {num} of the {code_name} in plain English.",
    "What obligations does Article {num} of the {code_name} establish?",
    "Explain the legal significance of Article {num} ({section}) of the {code_name}.",
    "Translate and explain Article {num} of the {code_name} for a non-specialist.",
    "What rights does Article {num} of the {code_name} confer?",
]

def fetch_code(code_id: str) -> dict:
    url = f"{LEGI_RAW_BASE}{code_id}.json"
    print(f"Fetching {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def extract_articles(node: dict, code_name: str, breadcrumb: list | None = None) -> list[dict]:
    breadcrumb = breadcrumb or []
    articles = []
    node_type = node.get("type", "")
    data = node.get("data", {})
    title = data.get("titre", "") or data.get("num", "")

    if node_type == "article":
        text = data.get("texte", "").strip()
        num = data.get("num", "")
        if text and len(text) > 50:
            articles.append({
                "code_name": code_name,
                "article_id": node.get("id", ""),
                "article_num": num,
                "section_path": " > ".join(breadcrumb),
                "text": re.sub(r"\s+", " ", text),
            })
    else:
        child_bc = breadcrumb + [title] if title else breadcrumb
        for child in node.get("children", []):
            articles.extend(extract_articles(child, code_name, child_bc))
    return articles

def build_alpaca_sample(article: dict, idx: int) -> dict:
    tmpl = INSTRUCTION_TEMPLATES[idx % len(INSTRUCTION_TEMPLATES)]
    section = article["section_path"].split(" > ")[-1] if article["section_path"] else "General"
    
    instruction = tmpl.format(
        num=article["article_num"], 
        code_name=article["code_name"],
        section=section
    )
    
    # In Alpaca, the 'output' is the ground truth response
    output = (
        f"**{article['code_name']} - Article {article['article_num']}**\n"
        f"Path: {article['section_path']}\n\n"
        f"{article['text']}"
    )
    
    # Formatting the full prompt for models that expect a single 'text' block
    full_text = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n\n"
        f"### Response:\n{output}"
    )
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "text": full_text
    }

def main():
    random.seed(42)
    all_articles = []

    for name, code_id in CODES.items():
        try:
            raw_data = fetch_code(code_id)
            code_articles = extract_articles(raw_data, code_name=name, breadcrumb=[name])
            print(f"  -> Extracted {len(code_articles):,} articles from {name}")
            all_articles.extend(code_articles)
        except Exception as e:
            print(f"  [!] Failed processing {name}: {e}")

    print(f"\nTotal articles: {len(all_articles):,}")

    samples = [build_alpaca_sample(a, i) for i, a in enumerate(all_articles)]
    random.shuffle(samples)

    split_idx = int(0.95 * len(samples))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Alpaca dataset created. Train: {len(train_samples)} | Eval: {len(eval_samples)}")

if __name__ == "__main__":
    main()