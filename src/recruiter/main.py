"""Main entrypoint for training the question generator."""
import os
import ray
import hydra
from pathlib import Path
from omegaconf import DictConfig
from typing import Optional

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_gym.envs import register

from .env import QuestionGenEnv, QuestionGenEnvConfig


# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Module-level metrics logger (set via environment variable for Ray workers)
_metrics_logger: Optional["MetricsLogger"] = None


def get_metrics_logger() -> Optional["MetricsLogger"]:
    """Get the global metrics logger if enabled."""
    global _metrics_logger
    if _metrics_logger is None and os.environ.get("METRICS_JSON_ENABLED") == "1":
        from .metrics_logger import MetricsLogger
        output_dir = os.environ.get("METRICS_JSON_DIR", None)
        _metrics_logger = MetricsLogger(output_dir=output_dir)
        print(f"Metrics logging enabled: {_metrics_logger.path}")
    return _metrics_logger


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
    Optionally logs rewards to metrics JSON.
    """
    _step_count = 0  # Class-level counter for tracking steps

    def __init__(self, **kwargs):
        super().__init__(QuestionGenEnvConfig())

    def step(self, action: str):
        """Override step to optionally log rewards."""
        result = super().step(action)

        # Log reward if metrics logging is enabled
        logger = get_metrics_logger()
        if logger is not None:
            QuestionGenEnvWrapper._step_count += 1
            # Log every N steps to avoid excessive writes (N=10)
            if QuestionGenEnvWrapper._step_count % 10 == 0:
                logger.log_step(
                    step=QuestionGenEnvWrapper._step_count,
                    reward=result.reward,
                )

        return result


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig, metrics_dir: Optional[str] = None):
    # Enable metrics logging in this Ray worker
    if metrics_dir:
        os.environ["METRICS_JSON_ENABLED"] = "1"
        os.environ["METRICS_JSON_DIR"] = metrics_dir

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

    # Determine metrics directory if logging is enabled
    metrics_dir = None
    if cfg.get("trainer", {}).get("metrics_json", False):
        # Use hydra output dir for metrics
        output_dir = Path.cwd()  # Hydra changes cwd to output dir
        metrics_dir = str(output_dir / "metrics")
        print(f"Metrics JSON logging enabled: {metrics_dir}")

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg, metrics_dir))


if __name__ == "__main__":
    main()
