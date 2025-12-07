"""Main entrypoint for training the question generator."""
import ray
import hydra
from omegaconf import DictConfig
import json

from skyrl_train.experiments.base_ppo_exp import BasePPOExp
from skyrl_train.utils.config import validate_cfg
from skyrl_train.utils.ray_utils import initialize_ray
from skyrl_gym.envs import register

from .env import QuestionGenEnv, QuestionGenEnvConfig


class QuestionGenEnvWrapper(QuestionGenEnv):
    """
    Wrapper that initializes from the dataset row.
    SkyRL passes the dataset row to the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(QuestionGenEnvConfig())
        self._role_json = kwargs.get("role_json", None)
        if self._role_json:
            role = json.loads(self._role_json)
            self.set_role(role)


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
