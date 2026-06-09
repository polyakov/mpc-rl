"""Phase-based gait MDP utilities for THEMIS."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

class phase:
    """Gait-clock observation that freezes to stance phase when standing."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        env._gait_phase = torch.zeros(env.num_envs, device=env.device)
        env._was_standing = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._prev_ep_len = env.episode_length_buf.clone()

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        period: float,
        command_name: str = "twist",
        command_threshold: float = 0.1,
        stance_decay: float = 0.95,
    ) -> torch.Tensor:
        reset_mask = env.episode_length_buf < self._prev_ep_len
        env._gait_phase[reset_mask] = 0.0
        env._was_standing[reset_mask] = True
        self._prev_ep_len = env.episode_length_buf.clone()

        cmd = env.command_manager.get_command(command_name)
        assert cmd is not None
        cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        moving = cmd_mag > command_threshold

        just_started = moving & env._was_standing
        if just_started.any():
            vy = cmd[just_started, 1]
            env._gait_phase[just_started] = torch.where(
                vy < 0,
                torch.full_like(vy, math.pi),
                torch.zeros_like(vy),
            )
        env._was_standing = ~moving

        dphi = 2.0 * math.pi * env.step_dt / period
        env._gait_phase[moving] += dphi
        env._gait_phase[~moving] *= stance_decay

        return torch.stack(
            [env._gait_phase.sin(), env._gait_phase.cos()], dim=-1
        )

def phase_readout(
    env: ManagerBasedRlEnv,
    period: float = 0.0,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    stance_decay: float = 0.95,
) -> torch.Tensor:
    """Read the accumulated gait phase (set by :class:`phase`)."""
    phi = getattr(
        env, "_gait_phase", torch.zeros(env.num_envs, device=env.device)
    )
    return torch.stack([phi.sin(), phi.cos()], dim=-1)

def feet_gait(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period: float,
    offset: list[float],
    threshold: float,
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Phase-locked gait reward."""
    if hasattr(env, "_gait_phase"):
        phi_base = env._gait_phase
    else:
        t = env.episode_length_buf * env.step_dt
        phi_base = 2.0 * math.pi * t / period

    offsets_t = torch.tensor(offset, device=env.device, dtype=torch.float32)
    phi = (
        phi_base.unsqueeze(1)
        + 2.0 * math.pi * offsets_t.unsqueeze(0)
    )
    desired_contact = (phi.sin() < threshold).float()

    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None, (
        f"ContactSensor '{sensor_name}' has no 'found' data"
    )
    actual_contact = (sensor.data.found > 0).float()

    reward = torch.mean(
        desired_contact * actual_contact
        + (1.0 - desired_contact) * (1.0 - actual_contact),
        dim=-1,
    )

    cmd = env.command_manager.get_command(command_name)
    if cmd is not None:
        cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        reward *= (cmd_mag > command_threshold).float()

    return reward

def stand_still(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalise joint drift from the default pose during zero-velocity commands."""
    asset = env.scene[asset_cfg.name]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(current_pos - default_pos), dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if cmd is not None:
        cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        error *= (cmd_mag < command_threshold).float()

    return error

def stand_still_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalise joint velocities during zero-velocity commands."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_penalty = torch.sum(torch.square(joint_vel), dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if cmd is not None:
        cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        vel_penalty *= (cmd_mag < command_threshold).float()

    return vel_penalty

class apply_hand_push_force:
    """Apply continuous push reaction forces to robot hands."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset = env.scene[cfg.params["asset_cfg"].name]
        self._body_ids = cfg.params["asset_cfg"].body_ids
        self._num_envs = env.num_envs
        self._device = env.device
        self._step_dt = env.step_dt
        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._asset.num_bodies
        )

        env._hand_push_force_scale = 0.0

        env._hand_push_force_mag = torch.zeros(
            env.num_envs, device=env.device
        )
        self._force_body = torch.zeros(
            env.num_envs, 3, device=env.device
        )
        hold_s = cfg.params.get("hold_duration_s", (5.0, 15.0))
        self._hold_range_steps = (
            int(hold_s[0] / self._step_dt),
            int(hold_s[1] / self._step_dt),
        )
        self._hold_remaining = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
        self._prev_ep_len = env.episode_length_buf.clone()

        all_ids = torch.arange(env.num_envs, device=env.device)
        self._resample(
            env,
            all_ids,
            cfg.params["max_normal_force"],
            cfg.params["friction_coeff"],
        )

    def _resample(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        max_normal_force: float,
        friction_coeff: float,
    ) -> None:
        n = len(env_ids)
        scale = getattr(env, "_hand_push_force_scale", 1.0)
        f_n = torch.rand(n, device=self._device) * max_normal_force * scale
        fy = (
            (2 * torch.rand(n, device=self._device) - 1)
            * friction_coeff
            * f_n
        )
        fz = (
            (2 * torch.rand(n, device=self._device) - 1)
            * friction_coeff
            * f_n
        )
        env._hand_push_force_mag[env_ids] = f_n
        self._force_body[env_ids, 0] = -f_n
        self._force_body[env_ids, 1] = fy
        self._force_body[env_ids, 2] = fz

        lo, hi = self._hold_range_steps
        self._hold_remaining[env_ids] = torch.randint(
            lo, hi + 1, (n,), device=self._device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        max_normal_force: float,
        friction_coeff: float,
        asset_cfg: SceneEntityCfg,
        hold_duration_s: tuple[float, float] = (5.0, 15.0),
        command_name: str = "twist",
        command_threshold: float = 0.1,
    ) -> None:
        del env_ids, asset_cfg, hold_duration_s

        reset_mask = env.episode_length_buf < self._prev_ep_len
        self._prev_ep_len = env.episode_length_buf.clone()

        self._hold_remaining -= 1
        expired = self._hold_remaining <= 0

        need_resample = reset_mask | expired
        if need_resample.any():
            resample_ids = need_resample.nonzero(as_tuple=False).squeeze(-1)
            self._resample(env, resample_ids, max_normal_force, friction_coeff)

        cmd = env.command_manager.get_command(command_name)
        cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        standing = cmd_mag < command_threshold

        base_quat = self._asset.data.root_link_quat_w
        forces_world = quat_apply(base_quat, self._force_body)

        forces_world[standing] = 0.0
        env._hand_push_force_mag[standing] = 0.0

        forces = forces_world.unsqueeze(1).expand(
            -1, self._num_bodies, -1
        ).contiguous()
        torques = torch.zeros_like(forces)

        all_ids = torch.arange(self._num_envs, device=self._device)
        self._asset.write_external_wrench_to_sim(
            forces, torques, env_ids=all_ids, body_ids=self._body_ids
        )

def torso_pitch_tracking(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    max_force: float,
    max_pitch: float,
    std: float = 0.15,
) -> torch.Tensor:
    """Reward the torso for pitching forward proportionally to hand push force."""
    asset = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b

    pitch_actual = torch.atan2(-proj_grav[:, 0], -proj_grav[:, 2])

    force_mag = getattr(
        env, "_hand_push_force_mag",
        torch.zeros(env.num_envs, device=env.device),
    )
    pitch_desired = (force_mag / max_force).clamp(max=1.0) * max_pitch

    error = pitch_actual - pitch_desired
    return torch.exp(-error.square() / (std * std))

def shoulder_pitch_compensation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    max_force: float,
    max_pitch: float,
    std: float = 0.2,
) -> torch.Tensor:
    """Reward shoulder pitch joints for compensating torso forward pitch."""
    asset = env.scene[asset_cfg.name]
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    force_mag = getattr(
        env, "_hand_push_force_mag",
        torch.zeros(env.num_envs, device=env.device),
    )
    desired_pitch = (force_mag / max_force).clamp(max=1.0) * max_pitch

    target_pos = default_pos - desired_pitch.unsqueeze(-1)

    error = (current_pos - target_pos).square().sum(dim=-1)
    return torch.exp(-error / (std * std))

def push_force_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    force_stages: list[dict],
) -> torch.Tensor:
    """Gradually increase hand push force scale over training."""
    del env_ids
    scale = 0.0
    for stage in force_stages:
        if env.common_step_counter > stage["step"]:
            scale = stage["scale"]
    env._hand_push_force_scale = scale
    return torch.tensor([scale])
