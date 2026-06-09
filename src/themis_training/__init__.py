"""MPC-guided RL training tasks for the Westwood Robotics THEMIS humanoid.

Registers two tasks with mjlab:

* ``Mjlab-MPC-Guided-Locomotion-Themis`` — MPC-guided velocity locomotion.
  A centroidal QP-MPC runs in parallel with the policy at the policy rate and
  supplies CoM / angular-momentum / GRF / foot-placement reference targets that
  shape the reward (solver: batched JAX PiMPC).

* ``Mjlab-MPC-Guided-Loco-manipulation-Themis`` — MPC-guided
  loco-manipulation. Extends the locomotion task with a box-pushing scene and
  swaps the locomotion-only MPC for the loco-manipulation MPC that also solves
  for hand push forces (solver: batched PyTorch PiMPC).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  themis_loco_manip_mpc_push_box_flat_env_cfg,
  themis_mpc_grf_v2_flat_env_cfg,
)
from .rl_cfg import themis_ppo_runner_cfg

# ── MPC-guided velocity locomotion ──────────────────────────────────────────
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Locomotion-Themis",
  env_cfg=themis_mpc_grf_v2_flat_env_cfg(),
  play_env_cfg=themis_mpc_grf_v2_flat_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# ── MPC-guided loco-manipulation (box pushing) ──────────────────────────────
register_mjlab_task(
  task_id="Mjlab-MPC-Guided-Loco-manipulation-Themis",
  env_cfg=themis_loco_manip_mpc_push_box_flat_env_cfg(),
  play_env_cfg=themis_loco_manip_mpc_push_box_flat_env_cfg(play=True),
  rl_cfg=themis_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
