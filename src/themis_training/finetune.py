"""Fine-tune wrapper for mjlab training with a mismatched checkpoint.

Mirrors ``mjlab.scripts.train`` one-to-one, except the checkpoint load call
uses ``strict=False`` so critic layers whose input dim changed (e.g.
privileged obs added by the fine-tune env) are re-initialised instead of
raising a size-mismatch.  Actor weights load fully because the actor input
shape is unchanged.

Usage:
  python -m themis_training.finetune Mjlab-Velocity-PushBox-Flat-Themis \\
      --agent.resume \\
      --agent.load-run=<loco_run_dir_name> \\
      --agent.load-checkpoint=model_29999.pt

Any other ``--agent.*`` / ``--env.*`` flags accepted by ``mjlab.scripts.train``
work here verbatim.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import list_tasks, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


def _filter_shape_mismatch(
  state_dict: dict[str, torch.Tensor],
  target_module: torch.nn.Module,
  label: str,
) -> dict[str, torch.Tensor]:
  """Drop keys whose shape differs from the target module's parameters.

  PyTorch's ``strict=False`` ignores missing/unexpected keys but still raises
  on shape mismatches for matching keys.  For warm-starting a fine-tune where
  a first linear layer's input dim changed (e.g. new privileged obs), we want
  to drop just that key so the layer stays randomly initialised and every
  other layer loads normally.
  """
  target_sd = target_module.state_dict()
  filtered: dict[str, torch.Tensor] = {}
  dropped: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
  for key, value in state_dict.items():
    if key in target_sd and target_sd[key].shape != value.shape:
      dropped.append((key, tuple(value.shape), tuple(target_sd[key].shape)))
      continue
    filtered[key] = value
  if dropped:
    print(f"[INFO] {label}: re-initialising {len(dropped)} shape-mismatched tensor(s):")
    for key, src_shape, dst_shape in dropped:
      print(f"       {key}: checkpoint {src_shape} -> current {dst_shape}")
  return filtered


def _load_checkpoint_loose(runner: MjlabOnPolicyRunner, path: str) -> None:
  """Reproduce mjlab's runner.load but filter shape mismatches per module.

  Mirrors ``mjlab.rl.runner.MjlabOnPolicyRunner.load`` (legacy key migration,
  ``std`` / ``log_std`` → distribution params, ``env_state`` restore) and
  calls ``load_state_dict`` on each module with ``strict=False`` AFTER
  dropping any keys whose shapes don't match.  Optimizer state is skipped
  because parameter shapes may have changed.
  """
  loaded_dict = torch.load(path, map_location="cpu", weights_only=False)

  # Legacy ``model_state_dict`` → split actor/critic. Copied from
  # MjlabOnPolicyRunner.load so old checkpoints still work.
  if "model_state_dict" in loaded_dict:
    print(f"[INFO] Detected legacy checkpoint at {path}. Migrating to new format...")
    model_state_dict = loaded_dict.pop("model_state_dict")
    actor_state_dict: dict[str, torch.Tensor] = {}
    critic_state_dict: dict[str, torch.Tensor] = {}
    for key, value in model_state_dict.items():
      if key.startswith("actor."):
        actor_state_dict[key.replace("actor.", "mlp.")] = value
      elif key.startswith("actor_obs_normalizer."):
        actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
      elif key in ("std", "log_std"):
        actor_state_dict[key] = value
      if key.startswith("critic."):
        critic_state_dict[key.replace("critic.", "mlp.")] = value
      elif key.startswith("critic_obs_normalizer."):
        critic_state_dict[key.replace("critic_obs_normalizer.", "obs_normalizer.")] = value
    loaded_dict["actor_state_dict"] = actor_state_dict
    loaded_dict["critic_state_dict"] = critic_state_dict

  # rsl-rl 4.x → 5.x actor distribution key rename.
  actor_sd = loaded_dict.get("actor_state_dict", {})
  if "std" in actor_sd:
    actor_sd["distribution.std_param"] = actor_sd.pop("std")
  if "log_std" in actor_sd:
    actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

  actor = runner.alg.actor
  critic = runner.alg.critic

  filtered_actor = _filter_shape_mismatch(actor_sd, actor, "actor")
  filtered_critic = _filter_shape_mismatch(
    loaded_dict.get("critic_state_dict", {}), critic, "critic"
  )

  actor.load_state_dict(filtered_actor, strict=False)
  critic.load_state_dict(filtered_critic, strict=False)

  # Optimizer state references parameter IDs that may have been re-initialised;
  # skip it so momentum doesn't carry stale statistics.
  print("[INFO] Skipping optimizer state (fresh momentum for fine-tune).")

  # Preserve curriculum step counter if present, as mjlab's runner.load does.
  infos = loaded_dict.get("infos")
  if infos and "env_state" in infos:
    runner.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]


def run_finetune(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()
  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Fine-tuning with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either --registry-name or "
        "--env.commands.motion.motion-file."
      )

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )
  else:
    print(
      "[WARN] finetune.py invoked without --agent.resume; nothing will be "
      "loaded. Use mjlab.scripts.train for fresh training."
    )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(
      f"[INFO]: Loading model checkpoint (loose, shape-filtered) from: {resume_path}"
    )
    _load_checkpoint_loose(runner, str(resume_path))

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_finetune(task_id: str, args: TrainConfig | None = None) -> None:
  args = args or TrainConfig.from_task(task_id)

  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_finetune(task_id, args, log_dir)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)

    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching fine-tune with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_finetune, task_id, args, log_dir)


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import themis_training  # noqa: F401  (registers THEMIS tasks)

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_finetune(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
