"""Main entrypoint for training the question generator."""
import ray
import hydra
from pathlib import Path
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_gym.envs import register

from .env import QuestionGenEnv, QuestionGenEnvConfig


# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def ensure_data_formatted():
    """Format prompts if needed before training starts.

    Automatically runs formatting if:
    - Processed data doesn't exist, OR
    - Raw data is newer than processed data, OR
    - prompts.py is newer than processed data
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    prompts_file = Path(__file__).parent / "prompts.py"

    raw_train = raw_dir / "train.parquet"
    processed_train = processed_dir / "train.parquet"

    # Check if raw data exists
    if not raw_train.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_dir}\n"
            f"Run: python scripts/prepare_dataset.py"
        )

    # Check if we need to format
    needs_format = False

    if not processed_train.exists():
        print("Processed data not found - formatting prompts...")
        needs_format = True
    elif raw_train.stat().st_mtime > processed_train.stat().st_mtime:
        print("Raw data updated - reformatting prompts...")
        needs_format = True
    elif prompts_file.exists() and prompts_file.stat().st_mtime > processed_train.stat().st_mtime:
        print("prompts.py updated - reformatting prompts...")
        needs_format = True

    if needs_format:
        import pandas as pd
        from .prompts import format_prompt

        processed_dir.mkdir(parents=True, exist_ok=True)

        for parquet_file in raw_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            df['prompt'] = df['role_json'].apply(format_prompt)
            output_path = processed_dir / parquet_file.name
            df.to_parquet(output_path, index=False)
            print(f"  Formatted {len(df)} examples -> {output_path.name}")

        print("Prompt formatting complete!\n")


class QuestionGenEnvWrapper(QuestionGenEnv):
    """
    Wrapper for SkyRL environment registration.
    SkyRL passes the prompt to init(), not __init__.
    """

    def __init__(self, **kwargs):
        super().__init__(QuestionGenEnvConfig())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register our custom environment
    register(
        id="question-gen",
        entry_point="src.recruiter.main:QuestionGenEnvWrapper",
    )

    # Run training
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Auto-format prompts if needed (before Ray starts)
    ensure_data_formatted()

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
