"""Modular JSON metrics logger for training."""
import json
from pathlib import Path
from typing import Optional
from datetime import datetime


class MetricsLogger:
    """Writes training metrics to JSON file. Standalone module."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the metrics logger.

        Args:
            output_dir: Directory to write metrics.json. If None, creates
                       results/training_<timestamp>/
        """
        if output_dir is None:
            output_dir = f"results/training_{datetime.now():%Y%m%d_%H%M%S}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.json"
        self.metrics = []

    def log_step(self, step: int, reward: float, **kwargs):
        """Log metrics for a training step.

        Args:
            step: Training step number
            reward: Average reward for this step
            **kwargs: Additional metrics (loss, kl, etc.)
        """
        entry = {"step": step, "reward": reward, **kwargs}
        self.metrics.append(entry)
        self._write()

    def log_eval(self, step: int, eval_reward: float, **kwargs):
        """Log evaluation metrics.

        Args:
            step: Training step when eval was run
            eval_reward: Average eval reward
            **kwargs: Additional eval metrics
        """
        # Find if we already have an entry for this step
        for entry in self.metrics:
            if entry["step"] == step:
                entry["eval_reward"] = eval_reward
                entry.update(kwargs)
                self._write()
                return
        # Otherwise create new entry
        entry = {"step": step, "eval_reward": eval_reward, **kwargs}
        self.metrics.append(entry)
        self._write()

    def _write(self):
        """Write metrics to JSON file."""
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    @property
    def path(self) -> Path:
        """Return path to metrics file."""
        return self.metrics_path
