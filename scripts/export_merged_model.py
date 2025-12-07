"""
Export script to merge LoRA adapters from SkyRL checkpoint with base model
and save as a standalone HuggingFace model.

Usage:
    # Basic export with verification
    python scripts/export_merged_model.py \
        --checkpoint checkpoints/global_step_100 \
        --output exports/qwen3-4b-question-gen \
        --verify

    # Push to HuggingFace Hub
    python scripts/export_merged_model.py \
        --checkpoint checkpoints/global_step_100 \
        --output exports/qwen3-4b-question-gen \
        --push_to_hub \
        --hub_repo username/model-name

    # Push to Hub AND create inference endpoint (T4 GPU, vLLM v0.8.5)
    python scripts/export_merged_model.py \
        --checkpoint checkpoints/global_step_100 \
        --output exports/qwen3-4b-question-gen \
        --push_to_hub \
        --hub_repo username/model-name \
        --create_endpoint

    # Create endpoint with L4 GPU instead of T4
    python scripts/export_merged_model.py \
        --checkpoint checkpoints/global_step_100 \
        --output exports/qwen3-4b-question-gen \
        --push_to_hub \
        --create_endpoint \
        --instance_type nvidia-l4

Environment Variables:
    HF_TOKEN: HuggingFace API token for authentication
    HF_REPO: Default repository name (optional, can be overridden with --hub_repo)

Notes:
    - Default max_model_len is 8192 (vLLM compatible, fits on T4)
    - Inference endpoints use vLLM v0.8.5 (stable for T4/older GPUs)
    - Endpoints are public with scale-to-zero after 1 hour by default
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default max model length for vLLM compatibility
# Qwen3 base models have 262144, which is too large for most GPUs
DEFAULT_MAX_MODEL_LEN = 8192

load_dotenv()


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
    max_model_len: int | None = None,
) -> None:
    """Save merged model in HuggingFace format with safetensors."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path}")

    # Save model weights as safetensors
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Update config.json with max_model_len for vLLM compatibility
    if max_model_len is not None:
        config_path = output_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        original_len = config.get("max_position_embeddings", "unknown")
        config["max_position_embeddings"] = max_model_len

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"  - Updated max_position_embeddings: {original_len} -> {max_model_len}")

    print("Model saved successfully!")
    print(f"  - Config: {output_dir / 'config.json'}")
    print(f"  - Weights: {output_dir / 'model.safetensors'}")
    print(f"  - Tokenizer: {output_dir / 'tokenizer.json'}")


def generate_model_card(output_path: str, base_model: str, repo_id: str | None = None) -> None:
    """Generate a README.md model card."""
    model_name = repo_id.split("/")[-1] if repo_id else Path(output_path).name

    readme_content = f"""---
license: apache-2.0
library_name: transformers
base_model: {base_model}
tags:
  - question-generation
  - rl
  - grpo
  - lora
pipeline_tag: text-generation
---

# {model_name}

Fine-tuned model for generating technical screening questions, trained using GRPO (Group Relative Policy Optimization) with LoRA adapters.

## Base Model

- **Base**: [{base_model}](https://huggingface.co/{base_model})
- **Training**: LoRA fine-tuning with RL (GRPO algorithm)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id or model_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id or model_name}")

prompt = "Generate a technical screening question for a senior backend engineer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or with vLLM for faster inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo_id or model_name}")
outputs = llm.generate(["Generate a technical screening question for a senior backend engineer:"], SamplingParams(max_tokens=256))
print(outputs[0].outputs[0].text)
```
"""

    readme_path = Path(output_path) / "README.md"
    readme_path.write_text(readme_content)
    print(f"  - Model card: {readme_path}")


def push_to_hub(output_path: str, repo_id: str, max_retries: int = 3) -> str:
    """Push the exported model to HuggingFace Hub."""
    print(f"\n=== Pushing to HuggingFace Hub ===")

    # Authenticate with HF
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Authenticating with HF_TOKEN environment variable...")
        login(token=hf_token)
    else:
        print("No HF_TOKEN found, using cached credentials...")

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/checking repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload with retry logic for race conditions
    print(f"Uploading to: https://huggingface.co/{repo_id}")
    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=output_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload RL-trained question generation model",
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                print(f"Upload failed, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nModel uploaded successfully!")
    print(f"View at: {url}")
    return url


def create_inference_endpoint(
    repo_id: str,
    endpoint_name: str | None = None,
    instance_type: str = "nvidia-t4",
    region: str = "us-east-1",
    vendor: str = "aws",
    min_replica: int = 0,
    max_replica: int = 1,
    scale_to_zero_timeout: int = 60,
    public: bool = True,
) -> str:
    """
    Create a HuggingFace Inference Endpoint with vLLM v0.8.5 (stable for T4 GPUs).

    Args:
        repo_id: HuggingFace model repository (e.g., "username/model-name")
        endpoint_name: Name for the endpoint (defaults to model name)
        instance_type: GPU type - "nvidia-t4", "nvidia-l4", "nvidia-a10g"
        region: Cloud region (default: us-east-1)
        vendor: Cloud provider - "aws", "gcp", "azure"
        min_replica: Minimum replicas (0 for scale-to-zero)
        max_replica: Maximum replicas
        scale_to_zero_timeout: Minutes before scaling to zero (default: 60)
        public: Whether endpoint is publicly accessible (default: True)

    Returns:
        Endpoint URL
    """
    from huggingface_hub import create_inference_endpoint

    print(f"\n=== Creating Inference Endpoint ===")

    # Authenticate
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable required for endpoint creation")

    login(token=hf_token)

    # Generate endpoint name from repo if not provided
    if endpoint_name is None:
        endpoint_name = repo_id.split("/")[-1].lower().replace("_", "-")[:32]

    print(f"Endpoint name: {endpoint_name}")
    print(f"Model: {repo_id}")
    print(f"Instance: {instance_type} (1x GPU, 16GB)")
    print(f"Region: {vendor}/{region}")
    print(f"vLLM Engine: vllm/vllm-openai:v0.8.5")
    print(f"Access: {'public' if public else 'protected'}")
    print(f"Scale-to-zero: After {scale_to_zero_timeout} minutes")

    try:
        endpoint = create_inference_endpoint(
            name=endpoint_name,
            repository=repo_id,
            framework="pytorch",
            task="text-generation",
            accelerator="gpu",
            instance_type=instance_type,
            instance_size="x1",
            region=region,
            vendor=vendor,
            min_replica=min_replica,
            max_replica=max_replica,
            scale_to_zero_timeout=scale_to_zero_timeout,
            type="public" if public else "protected",
            custom_image={
                "health_route": "/health",
                "url": "vllm/vllm-openai:v0.8.5",
                "port": 8000,
            },
            token=hf_token,
        )

        print(f"\nEndpoint created successfully!")
        print(f"Status: {endpoint.status}")
        print(f"URL: {endpoint.url}")
        print(f"\nNote: Endpoint may take a few minutes to start.")
        print(f"Monitor at: https://ui.endpoints.huggingface.co/{endpoint.namespace}/{endpoint_name}")

        return endpoint.url

    except Exception as e:
        print(f"\nFailed to create endpoint: {e}")
        print("\nYou can manually create the endpoint at:")
        print(f"  https://ui.endpoints.huggingface.co/new?repository={repo_id}")
        print("\nRecommended settings:")
        print(f"  - Instance: {instance_type} (1x GPU)")
        print(f"  - vLLM version: vllm/vllm-openai:v0.8.5")
        print(f"  - Region: {vendor}/{region}")
        raise


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
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the exported model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="HuggingFace Hub repository (default: HF_REPO env var)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"Max model length for vLLM compatibility (default: {DEFAULT_MAX_MODEL_LEN}). "
        "Set to 0 to keep original value from base model.",
    )
    parser.add_argument(
        "--create_endpoint",
        action="store_true",
        help="Create a HuggingFace Inference Endpoint after pushing to hub",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        default=None,
        help="Custom name for the inference endpoint (default: derived from model name)",
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        default="nvidia-t4",
        choices=["nvidia-t4", "nvidia-l4", "nvidia-a10g", "nvidia-a100"],
        help="GPU instance type for the endpoint (default: nvidia-t4)",
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
    max_model_len = args.max_model_len if args.max_model_len > 0 else None
    save_merged_model(merged_model, tokenizer, args.output, max_model_len=max_model_len)

    # Verify if requested
    if args.verify:
        verify_export(args.output, args.test_prompt)

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        hub_repo = args.hub_repo or os.getenv("HF_REPO")
        if not hub_repo:
            parser.error(
                "--push_to_hub requires --hub_repo or HF_REPO environment variable"
            )
        # Generate model card before uploading
        generate_model_card(args.output, args.base_model, hub_repo)
        push_to_hub(args.output, hub_repo)

        # Create inference endpoint if requested
        if args.create_endpoint:
            endpoint_url = create_inference_endpoint(
                repo_id=hub_repo,
                endpoint_name=args.endpoint_name,
                instance_type=args.instance_type,
            )
            print(f"\nEndpoint URL: {endpoint_url}")

    elif args.create_endpoint:
        parser.error("--create_endpoint requires --push_to_hub")


if __name__ == "__main__":
    main()
