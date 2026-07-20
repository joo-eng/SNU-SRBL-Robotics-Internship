from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from bhl_env_cfg import bhl_flat_env_cfg, bhl_rough_env_cfg
from bhl_rl_cfg import bhl_ppo_runner_cfg

register_mjlab_task(
  task_id="BHL-Flat",
  env_cfg=bhl_flat_env_cfg(),
  play_env_cfg=bhl_flat_env_cfg(play=True),
  rl_cfg=bhl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="BHL-Rough",
  env_cfg=bhl_rough_env_cfg(),
  play_env_cfg=bhl_rough_env_cfg(play=True),
  rl_cfg=bhl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
