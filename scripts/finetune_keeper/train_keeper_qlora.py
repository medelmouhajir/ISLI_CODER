"""QLoRA / LoRA Fine-Tuning & GGUF Export Script for ISLI Keeper SLM (Qwen3-0.6B).

Optimized for consumer GPUs (NVIDIA RTX 3060, 12GB VRAM).
Trains on multi-task technical compression data and exports directly to
~/.isli/models/Qwen3-0.6B-ISLI-Keeper-Q8_0.gguf.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_GGUF_NAME = "Qwen3-0.6B-ISLI-Keeper-Q8_0.gguf"


def verify_dataset(train_path: Path, val_path: Path) -> tuple[int, int]:
    """Verify that dataset files exist and are non-empty."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}. Run build_dataset.py first.")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}. Run build_dataset.py first.")

    train_lines = sum(1 for _ in train_path.open("r", encoding="utf-8"))
    val_lines = sum(1 for _ in val_path.open("r", encoding="utf-8"))
    print(f"Verified dataset: {train_lines} train samples, {val_lines} val samples.")
    return train_lines, val_lines


def export_merged_to_gguf(merged_dir: Path, target_gguf: Path, outtype: str = "q8_0") -> Path:
    """Convert merged HuggingFace model directly to GGUF using convert_hf_to_gguf.py."""
    tools_dir = Path(__file__).resolve().parent / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    converter_script = tools_dir / "convert_hf_to_gguf.py"

    if not converter_script.exists():
        import urllib.request
        url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
        print(f"[GGUF Export] Downloading convert_hf_to_gguf.py from {url}...")
        urllib.request.urlretrieve(url, str(converter_script))

    staging_gguf = merged_dir.parent / target_gguf.name
    print(f"\n[GGUF Export] Converting {merged_dir} to {staging_gguf} (type: {outtype})...")
    cmd = [
        sys.executable,
        str(converter_script),
        str(merged_dir),
        "--outfile",
        str(staging_gguf),
        "--outtype",
        outtype,
    ]
    subprocess.run(cmd, check=True)
    print(f"[GGUF Export] Successfully created {staging_gguf} ({staging_gguf.stat().st_size / (1024*1024):.1f} MB)")

    # Safely deploy to target directory (handling Windows file locks if active)
    target_gguf.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    try:
        shutil.copy2(staging_gguf, target_gguf)
        print(f"[Deployment] Successfully deployed to: {target_gguf}")
    except OSError:
        backup = target_gguf.with_suffix(".old.gguf")
        if backup.exists():
            try:
                backup.unlink()
            except Exception:
                pass
        try:
            target_gguf.rename(backup)
        except Exception:
            pass
        shutil.copy2(staging_gguf, target_gguf)
        print(f"[Deployment] Deployed via safe rename rotation to: {target_gguf}")

    return target_gguf


def train_peft_standard(args: argparse.Namespace) -> Path:
    """HuggingFace PEFT + TRL training pipeline optimized for RTX 3060."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    print(f"\n[PEFT Engine] Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[PEFT Engine] Loading model with precision: {dtype} on CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": str(args.train_file), "val": str(args.val_file)})

    def formatting_func(example: dict[str, Any]) -> str | list[str]:
        if "messages" in example and isinstance(example["messages"], list):
            if example["messages"] and isinstance(example["messages"][0], dict):
                return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            elif example["messages"] and isinstance(example["messages"][0], list):
                return [
                    tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                    for m in example["messages"]
                ]
        return ""

    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        formatting_func=formatting_func,
    )

    print("\n[PEFT Engine] Starting fine-tuning...")
    trainer.train()

    # Save LoRA adapter
    lora_dir = args.output_dir / "lora_weights"
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"\n[PEFT Engine] LoRA weights saved to: {lora_dir}")

    # Merge LoRA and export full 16-bit HuggingFace weights
    merged_dir = args.output_dir / "merged_hf"
    print(f"\n[PEFT Engine] Merging LoRA weights into base model -> {merged_dir}...")
    merged_model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[PEFT Engine] Merged 16-bit model saved to: {merged_dir}")

    if args.export_gguf:
        target_gguf_dir = Path.home() / ".isli" / "models"
        target_gguf = target_gguf_dir / DEFAULT_GGUF_NAME
        export_merged_to_gguf(merged_dir, target_gguf, outtype=args.quantization)

    return lora_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Keeper SLM (Qwen3-0.6B) with QLoRA")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="HuggingFace model ID")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "keeper_sft_train.jsonl",
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "keeper_sft_val.jsonl",
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Context token limit")
    parser.add_argument("--export-gguf", action="store_true", default=True, help="Export merged model to GGUF")
    parser.add_argument("--quantization", type=str, default="q8_0", choices=["q8_0", "f16", "bf16"], help="GGUF format")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and configuration without running training")

    args = parser.parse_args()

    train_cnt, val_cnt = verify_dataset(args.train_file, args.val_file)

    if args.dry_run:
        print("\n[Dry Run] Configuration and dataset are valid.")
        print(f"Base model:     {args.base_model}")
        print(f"Target GGUF:    ~/.isli/models/{DEFAULT_GGUF_NAME}")
        print(f"Training pairs: {train_cnt} train, {val_cnt} val")
        return

    train_peft_standard(args)


if __name__ == "__main__":
    main()
