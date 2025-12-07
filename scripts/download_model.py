"""Download RL-trained model from HuggingFace Hub to exports/ for online RL training."""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def main():
    hf_repo = os.getenv("HF_REPO", "ash256/qwen3-4b-question-gen")
    output_dir = Path("exports") / hf_repo.split("/")[-1]

    print(f"Downloading {hf_repo} to {output_dir}...")
    snapshot_download(
        repo_id=hf_repo,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {output_dir}")
    print(f"\nTo use for online RL, update train_config.yaml:")
    print(f'  trainer.policy.model.path: "{output_dir}"')


if __name__ == "__main__":
    main()
