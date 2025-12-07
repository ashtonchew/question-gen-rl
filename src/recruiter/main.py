"""Main entrypoint for training the question generator."""
import ray
import hydra
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_gym.envs import register

from .env import QuestionGenEnv, QuestionGenEnvConfig


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
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
