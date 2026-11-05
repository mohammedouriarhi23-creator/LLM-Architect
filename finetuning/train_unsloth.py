"""
Unsloth QLoRA Fine-Tuning & GGUF Export Script for LLMarchitect.

Target Low-VRAM (6-8GB) Base Models:
  - unsloth/Qwen2.5-3B-Instruct-bnb-4bit
  - unsloth/Llama-3.2-3B-Instruct-bnb-4bit

Workflow:
  1. Generate dataset via backend (`POST /api/dataset/generate`) or LM Studio synthetic pipeline.
  2. Run this script to fine-tune the 3B model on `finetuning/datasets/llm_advisor_train.jsonl`.
  3. Automatically exports a `Q4_K_M` GGUF model ready to load directly into LM Studio!

Usage:
  python finetuning/train_unsloth.py --base-model qwen-2.5-3b --export-gguf
"""

import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "qlora_3b_config.json"
DEFAULT_DATASET_PATH = BASE_DIR / "datasets" / "llm_advisor_train.jsonl"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def train(
    base_model_key: str = "qwen-2.5-3b",
    dataset_path: Path = DEFAULT_DATASET_PATH,
    max_steps: int | None = None,
    export_gguf: bool = True,
) -> None:
    cfg = load_config()
    model_map = cfg["supported_base_models"]
    model_name = model_map.get(base_model_key, base_model_key)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run the backend dataset generator first (/api/dataset/generate)."
        )

    print(f"[LLMarchitect] Loading base model: {model_name} (4-bit QLoRA for 6-8GB VRAM)")

    # Lazy import so dry-runs / config checks work on machines without a CUDA GPU
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt
    from datasets import load_dataset
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments

    max_seq_length = cfg["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg["load_in_4bit"],
    )

    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )

    chat_template_name = "qwen-2.5" if "qwen" in model_name.lower() else "llama-3.1"
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template_name)

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = standardize_sharegpt(raw_dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    train_cfg = cfg["training"]
    output_dir = BASE_DIR / "outputs" / "lora_adapter"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            warmup_steps=train_cfg["warmup_steps"],
            max_steps=max_steps or train_cfg["max_steps"],
            learning_rate=train_cfg["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=train_cfg["logging_steps"],
            optim=train_cfg["optim"],
            weight_decay=train_cfg["weight_decay"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            seed=train_cfg["seed"],
            output_dir=str(output_dir),
            report_to="none",
        ),
    )

    print("[LLMarchitect] Starting Unsloth QLoRA fine-tuning...")
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[LLMarchitect] Saved LoRA adapter to: {output_dir}")

    if export_gguf:
        gguf_dir = BASE_DIR / cfg["gguf_export"]["output_dir"]
        quant_method = cfg["gguf_export"]["quantization_method"]
        print(f"[LLMarchitect] Exporting GGUF ({quant_method}) for LM Studio to: {gguf_dir}")
        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method=quant_method,
            )
            print(
                "\n[LLMarchitect] GGUF Export Complete!\n"
                f"Import the `.gguf` file from `{gguf_dir}` into LM Studio and start the Local Server on port 1234."
            )
        except Exception as exc:
            print(f"[LLMarchitect] Note: GGUF conversion step message: {exc}")
            merged_dir = BASE_DIR / "outputs" / "merged_16bit"
            print(f"[LLMarchitect] Saving merged 16-bit model to {merged_dir} as fallback...")
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 3B LLM Advisor with Unsloth & export to LM Studio GGUF")
    parser.add_argument(
        "--base-model",
        default="qwen-2.5-3b",
        choices=["qwen-2.5-3b", "llama-3.2-3b"],
        help="Low-VRAM 3B base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to ShareGPT JSONL training file",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override training steps")
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        default=True,
        help="Export fine-tuned model to GGUF Q4_K_M for LM Studio",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and dataset without launching CUDA training",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.dry_run:
        print(f"[Dry Run] Base model resolved to: {cfg['supported_base_models'][args.base_model]}")
        print(f"[Dry Run] Dataset path: {args.dataset} (exists={args.dataset.exists()})")
        print(f"[Dry Run] LoRA r={cfg['lora']['r']}, GGUF quant={cfg['gguf_export']['quantization_method']}")
        return

    train(
        base_model_key=args.base_model,
        dataset_path=args.dataset,
        max_steps=args.max_steps,
        export_gguf=args.export_gguf,
    )


if __name__ == "__main__":
    main()
