"""
Export script to merge LoRA adapters from SkyRL checkpoint with base model
and save as a standalone HuggingFace model.

Usage:
    python scripts/export_merged_model.py \
        --checkpoint checkpoints/global_step_100 \
        --output exports/qwen3-4b-question-gen \
        --verify
"""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_skyrl_checkpoint(checkpoint_path: Path) -> tuple[dict, dict]:
    """
    Load SkyRL checkpoint and separate LoRA weights from base model weights.

    SkyRL FSDP checkpoints store the full model state dict with LoRA parameters
    named with 'lora_A' and 'lora_B' for each target module.

    Returns:
        Tuple of (base_state_dict, lora_state_dict)
    """
    model_state_path = checkpoint_path / "policy" / "model_state.pt"

    if not model_state_path.exists():
        raise FileNotFoundError(f"Model state not found at {model_state_path}")

    print(f"Loading checkpoint from {model_state_path}")
    state_dict = torch.load(model_state_path, map_location="cpu", weights_only=True)

    # Separate LoRA weights from base model weights
    lora_state_dict = {}
    base_state_dict = {}

    for key, value in state_dict.items():
        if "lora_" in key.lower():
            lora_state_dict[key] = value
        else:
            base_state_dict[key] = value

    print(f"Found {len(lora_state_dict)} LoRA parameters")
    print(f"Found {len(base_state_dict)} base model parameters")

    return base_state_dict, lora_state_dict


def load_from_lora_sync_path(
    base_model_path: str,
    lora_sync_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """
    Load LoRA from SkyRL's sync path if available.
    This path may contain PEFT-compatible adapter files.
    """
    lora_path = Path(lora_sync_path)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA sync path not found: {lora_sync_path}")

    # Check for adapter_config.json (PEFT format)
    if (lora_path / "adapter_config.json").exists():
        print(f"Loading PEFT adapter from {lora_sync_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, lora_sync_path)
        return peft_model.merge_and_unload()

    raise ValueError(f"No PEFT adapter_config.json found at {lora_sync_path}")


def load_and_merge_model(
    base_model_path: str,
    checkpoint_path: Path,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """
    Load base model, apply LoRA weights from checkpoint, and merge.

    SkyRL saves LoRA adapters in PEFT format at: checkpoint/policy/lora_adapter/
    """
    # Check for PEFT-format lora_adapter directory (SkyRL default)
    lora_adapter_path = checkpoint_path / "policy" / "lora_adapter"

    if lora_adapter_path.exists() and (lora_adapter_path / "adapter_config.json").exists():
        print(f"Found PEFT adapter at {lora_adapter_path}")
        print(f"Loading base model: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Loading LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        print("Merging LoRA weights with base model...")
        return peft_model.merge_and_unload()

    # Fallback: try loading from model state file
    model_state_path = checkpoint_path / "policy" / "model_state.pt"
    if not model_state_path.exists():
        # Try FSDP sharded format
        model_state_path = checkpoint_path / "policy" / "model_world_size_1_rank_0.pt"

    if not model_state_path.exists():
        raise FileNotFoundError(
            f"No LoRA adapter or model state found in {checkpoint_path}/policy/. "
            f"Expected either 'lora_adapter/' directory or 'model_state.pt' file."
        )

    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    base_state_dict, lora_state_dict = load_skyrl_checkpoint(checkpoint_path)

    if not lora_state_dict:
        print("Warning: No LoRA weights found in checkpoint. Returning base model.")
        return base_model

    # Infer LoRA rank from weights
    sample_key = next((k for k in lora_state_dict if "lora_A" in k), None)
    if sample_key:
        rank = lora_state_dict[sample_key].shape[0]
    else:
        rank = 8  # Default from train_config.yaml

    print(f"Inferred LoRA rank: {rank}")

    # Create LoRA config matching train_config.yaml
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=16,
        lora_dropout=0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply PEFT LoRA to base model
    print("Applying LoRA configuration to base model...")
    peft_model = get_peft_model(base_model, lora_config)

    # Load LoRA weights
    print("Loading LoRA weights into PEFT model...")
    peft_model.load_state_dict(lora_state_dict, strict=False)

    # Merge LoRA weights into base model
    print("Merging LoRA weights with base model...")
    merged_model = peft_model.merge_and_unload()

    return merged_model


def save_merged_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: str,
) -> None:
    """Save merged model in HuggingFace format with safetensors."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path}")

    # Save model weights as safetensors
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print("Model saved successfully!")
    print(f"  - Config: {output_dir / 'config.json'}")
    print(f"  - Weights: {output_dir / 'model.safetensors'}")
    print(f"  - Tokenizer: {output_dir / 'tokenizer.json'}")


def verify_export(output_path: str, test_prompt: str | None = None) -> bool:
    """Verify the exported model loads and generates correctly."""
    print("\n=== Verification Test ===")

    # Load with standard transformers
    model = AutoModelForCausalLM.from_pretrained(
        output_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(output_path)

    # Test generation
    if test_prompt is None:
        test_prompt = "Generate a technical screening question for a senior Python developer:"

    print(f"\nTest prompt: {test_prompt}")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{response}")
    print("\nVerification PASSED - Model loads and generates successfully!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters from SkyRL checkpoint and export as HuggingFace model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to SkyRL checkpoint directory (e.g., checkpoints/global_step_100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/merged_model",
        help="Output path for merged model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model to merge LoRA weights into",
    )
    parser.add_argument(
        "--lora_sync_path",
        type=str,
        default=None,
        help="Alternative: Load LoRA from sync path instead of checkpoint",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification test after export",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default=None,
        help="Custom prompt for verification test",
    )

    args = parser.parse_args()

    # Validate args
    if not args.checkpoint and not args.lora_sync_path:
        parser.error("Either --checkpoint or --lora_sync_path must be provided")

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Load tokenizer from base model
    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load and merge model
    if args.lora_sync_path:
        merged_model = load_from_lora_sync_path(
            args.base_model,
            args.lora_sync_path,
            torch_dtype=torch_dtype,
        )
    else:
        merged_model = load_and_merge_model(
            args.base_model,
            Path(args.checkpoint),
            torch_dtype=torch_dtype,
        )

    # Save merged model
    save_merged_model(merged_model, tokenizer, args.output)

    # Verify if requested
    if args.verify:
        verify_export(args.output, args.test_prompt)


if __name__ == "__main__":
    main()
