# Fine-Tuning LLMs on Légifrance with LoRA (MPS)

Companion code for the blog post **[Fine-Tuning LLMs: When Prompting Is Not Enough](https://mariazork.github.io/blog/2026-04-01-fine-tuning-llms-when-prompting-is-not-enough/)**.

Model artifacts are available at the [**link**](https://drive.google.com/drive/folders/1U-LID-Rxmq8KzBvdsB66s4ZBindhbCSI?usp=drive_link)

Fine-tunes LLM models on French legal articles from the [legi-data](https://github.com/SocialGouv/legi-data) open-data mirror of Légifrance, using LoRA + rsLoRA via Hugging Face PEFT. All code targets **Apple Silicon MPS** (also runs on CUDA and CPU).

## Quick Start

```bash
pip install -r requirements.txt

# 1. Prepare dataset (fetches Code du travail from GitHub raw)
python scripts/prepare_dataset.py

# 2. Train (MPS auto-detected)
python scripts/train_lora.py

# 3. Inference
python scripts/inference.py

# 4. Evaluate
python scripts/evaluate_model.py
```

## Key Design Choices for MPS

| Setting | Value | Reason |
|---|---|---|
| `dtype` | `float16` | bfloat16 not fully stable on MPS |
| `dataloader_pin_memory` | `False` | MPS does not support pinned memory |
| `optim` | `adamw_torch` | `paged_adamw_8bit` requires CUDA bitsandbytes |
| `bf16` | `False` | MPS limitation |
| `fp16` | `True` | Works on MPS with PyTorch ≥ 2.1 |
| `bitsandbytes` | not used | CUDA-only; QLoRA not available on MPS |

## References

- Hu et al. (2021). LoRA. arXiv:2106.09685
- Dettmers et al. (2023). QLoRA. arXiv:2305.14314
- Liu et al. (2024). DoRA. arXiv:2402.09353
- SocialGouv. legi-data. https://github.com/SocialGouv/legi-data
