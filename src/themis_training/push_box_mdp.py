"""Perceptive loco-manipulation MDP for the THEMIS push-box task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from torch import Tensor

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

_PUSH_MODE_ATTR: str = "_push_mode"

def _push_mask(env: "ManagerBasedRlEnv") -> Tensor:
    """Return the push-mode BoolTensor. All-True fallback if uninitialised."""
    mask = getattr(env, _PUSH_MODE_ATTR, None)
    if mask is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return mask

def init_push_mode(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    push_fraction: float = 2.0 / 3.0,
) -> None:
    """Assign a fixed walk-only / push mode to each env (startup event)."""
    del env_ids
    n_push = int(round(env.num_envs * push_fraction))
    mode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mode[:n_push] = True
    setattr(env, _PUSH_MODE_ATTR, mode)

class ModeGatedVelocityCommand(UniformVelocityCommand):
    """Velocity command that enforces forward-only for push-mode envs."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        mask = getattr(self._env, _PUSH_MODE_ATTR, None)
        if mask is None:
            return
        push_here = env_ids[mask[env_ids]]
        if len(push_here) == 0:
            return
        max_vx = max(0.0, float(self.cfg.ranges.lin_vel_x[1]))
        r = torch.empty(len(push_here), device=self.device)
        self.vel_command_b[push_here, 0] = r.uniform_(0.0, max_vx)
        self.vel_command_b[push_here, 1] = 0.0
        self.vel_command_b[push_here, 2] = 0.0

@dataclass(kw_only=True)
class ModeGatedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Drop-in replacement for :class:`UniformVelocityCommandCfg`."""

    def build(self, env: "ManagerBasedRlEnv") -> ModeGatedVelocityCommand:
        return ModeGatedVelocityCommand(self, env)

_BOX_MAX_HALF_W: float = 0.50
_BOX_MAX_HALF_D: float = 0.50
_BOX_MAX_HALF_H: float = 0.50

def get_push_box_spec() -> mujoco.MjSpec:
    """MuJoCo spec for the pushable box entity (fixed 1 m × 1 m × 1.3 m)."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
        name="box",
        pos=(0.0, 0.0, _BOX_MAX_HALF_H),
    )
    body.add_freejoint(name="box_joint")
    body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(_BOX_MAX_HALF_W, _BOX_MAX_HALF_D, _BOX_MAX_HALF_H),
        mass=5.0,
        rgba=(0.75, 0.55, 0.25, 1.0),
        friction=(1.0, 0.005, 0.0001),
    )
    return spec

def chest_point_cloud(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    max_range: float = 2.5,
    clamp_no_hit: bool = True,
) -> Tensor:
    """Flattened sparse point cloud in camera (sensor-site) frame."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    hit_pos_w = data.hit_pos_w
    distances = data.distances
    origin_w  = data.pos_w
    quat_w    = data.quat_w

    B, N, _ = hit_pos_w.shape

    rel_w = hit_pos_w - origin_w.unsqueeze(1)
    q_inv = quat_inv(quat_w)
    q_rep = q_inv.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)
    v_rep = rel_w.reshape(B * N, 3)
    rel_local = quat_apply_inverse(q_rep, v_rep).reshape(B, N, 3)

    if clamp_no_hit:
        miss = (distances < 0) | (distances > max_range)
        sentinel = torch.tensor(
            [0.0, 0.0, max_range], device=rel_local.device, dtype=rel_local.dtype
        )
        rel_local = torch.where(
            miss.unsqueeze(-1), sentinel.view(1, 1, 3).expand_as(rel_local), rel_local
        )

    return rel_local.reshape(B, N * 3)

def _box_state(env: "ManagerBasedRlEnv", box_name: str = "box"):
    """Return (box_pos_w, box_quat_w, box_lin_vel_w) for the push-box."""
    box = env.scene[box_name]
    return (
        box.data.root_link_pos_w,
        box.data.root_link_quat_w,
        box.data.root_link_lin_vel_w,
    )

def _robot_base_state(env: "ManagerBasedRlEnv", robot_name: str = "robot"):
    """Return (base_pos_w, base_quat_w)."""
    robot = env.scene[robot_name]
    return robot.data.root_link_pos_w, robot.data.root_link_quat_w

def box_pose_rel_priv(
    env: "ManagerBasedRlEnv",
    box_name: str = "box",
    robot_name: str = "robot",
) -> Tensor:
    """Box pose in robot-base frame: [dx, dy, dz, qw, qx, qy, qz]."""
    box_pos_w, box_quat_w, _ = _box_state(env, box_name)
    base_pos_w, base_quat_w = _robot_base_state(env, robot_name)

    delta_w = box_pos_w - base_pos_w
    delta_b = quat_apply_inverse(base_quat_w, delta_w)
    q_rel = quat_mul(quat_inv(base_quat_w), box_quat_w)
    return torch.cat([delta_b, q_rel], dim=-1)

def box_lin_vel_priv(
    env: "ManagerBasedRlEnv",
    box_name: str = "box",
    robot_name: str = "robot",
) -> Tensor:
    """Box linear velocity in robot-base frame."""
    _, _, box_v_w = _box_state(env, box_name)
    _, base_quat_w = _robot_base_state(env, robot_name)
    return quat_apply_inverse(base_quat_w, box_v_w)

def box_size_priv(
    env: "ManagerBasedRlEnv",
    box_name: str = "box",
    geom_name: str = "box_geom",
) -> Tensor:
    """Box half-extents (may vary per-env after geom_size randomisation)."""
    box = env.scene[box_name]
    local_geom_ids, _ = box.find_geoms(geom_name, preserve_order=True)
    assert len(local_geom_ids) == 1, f"Expected one geom for '{geom_name}'"
    gid = int(box.indexing.geom_ids[local_geom_ids[0]])
    size = env.sim.model.geom_size[:, gid, :]
    return size

def hand_box_contact_priv(
    env: "ManagerBasedRlEnv",
    lhand_sensor: str = "lhand_box_contact",
    rhand_sensor: str = "rhand_box_contact",
) -> Tensor:
    """Binary hand-box contact flags: [L_in_contact, R_in_contact]."""
    sl: ContactSensor = env.scene[lhand_sensor]
    sr: ContactSensor = env.scene[rhand_sensor]
    assert sl.data.found is not None and sr.data.found is not None
    l = (sl.data.found > 0).any(dim=-1, keepdim=False).float().unsqueeze(-1)
    r = (sr.data.found > 0).any(dim=-1, keepdim=False).float().unsqueeze(-1)
    return torch.cat([l, r], dim=-1)

def _hand_in_contact(env: "ManagerBasedRlEnv",
                     lhand_sensor: str,
                     rhand_sensor: str) -> tuple[Tensor, Tensor]:
    """Per-env booleans for L / R hand touching the box (any column)."""
    sl: ContactSensor = env.scene[lhand_sensor]
    sr: ContactSensor = env.scene[rhand_sensor]
    assert sl.data.found is not None and sr.data.found is not None
    l = (sl.data.found > 0).any(dim=-1)
    r = (sr.data.found > 0).any(dim=-1)
    return l, r

def hand_box_contact(
    env: "ManagerBasedRlEnv",
    lhand_sensor: str = "lhand_box_contact",
    rhand_sensor: str = "rhand_box_contact",
    both_hands_bonus: float = 0.5,
) -> Tensor:
    """Reward for contacting the box with one/both hands."""
    l, r = _hand_in_contact(env, lhand_sensor, rhand_sensor)
    both = (l & r).float()
    one  = (l ^ r).float()
    return (both + (1.0 - both_hands_bonus) * one) * _push_mask(env).float()

def push_velocity_match(
    env: "ManagerBasedRlEnv",
    command_name: str = "twist",
    box_name: str = "box",
    robot_name: str = "robot",
    lhand_sensor: str = "lhand_box_contact",
    rhand_sensor: str = "rhand_box_contact",
    sigma: float = 0.5,
) -> Tensor:
    """Reward the box being driven at the commanded forward velocity."""
    cmd = env.command_manager.get_command(command_name)
    if cmd is None:
        return torch.zeros(env.num_envs, device=env.device)

    vx_body_cmd = cmd[:, 0]

    _, _, box_v_w = _box_state(env, box_name)
    _, base_quat_w = _robot_base_state(env, robot_name)
    box_v_body = quat_apply_inverse(base_quat_w, box_v_w)
    vx_body_box = box_v_body[:, 0]

    err = (vx_body_box - vx_body_cmd).pow(2)
    gate_l, gate_r = _hand_in_contact(env, lhand_sensor, rhand_sensor)
    gate = (gate_l | gate_r).float() * _push_mask(env).float()

    return gate * torch.exp(-err / (sigma * sigma))

def box_com_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    box_name: str = "box",
    lhand_sensor: str = "lhand_box_contact",
    rhand_sensor: str = "rhand_box_contact",
    w_pos: float = 2.0,
    w_vel: float = 0.5,
    lookahead_fracs: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    lookahead_weights: tuple[float, ...] = (0.35, 0.25, 0.20, 0.12, 0.08),
) -> Tensor:
    """Reward box position and velocity tracking the MPC-planned box trajectory."""
    from themis_training.mpc_grf_mdp import LocoManipMPCCommand

    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoManipMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    box = env.scene[box_name]
    box_pos = box.data.root_link_pos_w
    box_vel = box.data.root_link_lin_vel_w

    box_ref_traj = term._vis_box_ref_traj
    N = term.cfg.mpc_horizon

    ref_vel_traj = (box_ref_traj[:, 1:, :] - box_ref_traj[:, :-1, :]) / term.cfg.mpc_dt

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        p_ref = (1 - alpha) * box_ref_traj[:, idx] + alpha * box_ref_traj[:, idx + 1]
        v_ref = (1 - alpha) * ref_vel_traj[:, idx] + alpha * ref_vel_traj[:, min(idx + 1, N - 1)]
        e_pos = box_pos - p_ref
        e_vel = box_vel - v_ref
        total = total + weight * (
            w_pos * e_pos[:, :2].pow(2).sum(dim=-1)
            + w_vel * e_vel[:, :2].pow(2).sum(dim=-1)
        )

    gate_l, gate_r = _hand_in_contact(env, lhand_sensor, rhand_sensor)
    gate = (gate_l | gate_r).float() * _push_mask(env).float()
    return gate * torch.exp(-total)

def robot_box_velocity_match(
    env: "ManagerBasedRlEnv",
    box_name: str = "box",
    robot_name: str = "robot",
    lhand_sensor: str = "lhand_box_contact",
    rhand_sensor: str = "rhand_box_contact",
    sigma: float = 0.3,
) -> Tensor:
    """Reward robot base linear velocity matching box linear velocity."""
    robot = env.scene[robot_name]
    _, base_quat_w = _robot_base_state(env, robot_name)
    robot_v_w = robot.data.root_link_lin_vel_w
    _, _, box_v_w = _box_state(env, box_name)
    robot_v_body = quat_apply_inverse(base_quat_w, robot_v_w)
    box_v_body = quat_apply_inverse(base_quat_w, box_v_w)
    err = (robot_v_body - box_v_body).pow(2).sum(dim=-1)

    gate_l, gate_r = _hand_in_contact(env, lhand_sensor, rhand_sensor)
    gate = (gate_l | gate_r).float() * _push_mask(env).float()
    return gate * torch.exp(-err / (sigma * sigma))

def robot_box_xy_distance_cost(
    env: "ManagerBasedRlEnv",
    box_name: str = "box",
    robot_name: str = "robot",
    target_distance: float = 0.5,
) -> Tensor:
    """Penalty for the robot drifting far from the box in the XY plane."""
    base_pos_w = env.scene[robot_name].data.root_link_pos_w
    box_pos_w = env.scene[box_name].data.root_link_pos_w
    dist = (box_pos_w[:, :2] - base_pos_w[:, :2]).norm(dim=-1)
    excess = (dist - target_distance).clamp(min=0.0)
    return excess * _push_mask(env).float()

def robot_box_yaw_cost(
    env: "ManagerBasedRlEnv",
    box_name: str = "box",
    robot_name: str = "robot",
) -> Tensor:
    """Penalty for robot-box yaw misalignment (push-mode only)."""
    _, base_quat_w = _robot_base_state(env, robot_name)
    box_quat_w = env.scene[box_name].data.root_link_quat_w
    yaw_r = yaw_quat(base_quat_w)
    yaw_b = yaw_quat(box_quat_w)
    q_rel = quat_mul(quat_inv(yaw_r), yaw_b)
    yaw_diff = 2.0 * torch.atan2(q_rel[:, 3], q_rel[:, 0]).abs()
    yaw_diff = torch.minimum(yaw_diff, 2.0 * torch.pi - yaw_diff)
    return yaw_diff * _push_mask(env).float()

def leg_box_collision_cost(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "leg_box_contact",
) -> Tensor:
    """Penalty when robot's legs / shins / thighs touch the box."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    hit = (sensor.data.found > 0).any(dim=-1).float()
    return hit * _push_mask(env).float()

class HandSlipPenalty:
    """Stateful penalty for losing hand-box contact mid-push."""

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv") -> None:
        self._env = env
        self._prev_in_contact: Tensor = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        lhand_sensor: str = "lhand_box_contact",
        rhand_sensor: str = "rhand_box_contact",
        box_name: str = "box",
        robot_name: str = "robot",
        reach_distance: float = 0.35,
        camera_site_name: str = "chest_camera",
    ) -> Tensor:
        l, r = _hand_in_contact(env, lhand_sensor, rhand_sensor)
        cur = l | r

        box = env.scene[box_name]
        robot = env.scene[robot_name]
        site_ids, _ = robot.find_sites(camera_site_name, preserve_order=True)
        assert len(site_ids) == 1
        cam_pos_w = robot.data.site_pos_w[:, int(site_ids[0]), :]
        box_pos_w = box.data.root_link_pos_w
        dist = (box_pos_w - cam_pos_w).norm(dim=-1)
        reachable = dist < reach_distance

        slip = self._prev_in_contact & reachable & (~cur) & _push_mask(env)
        self._prev_in_contact = cur.detach()

        return slip.float()

    def reset(self, env_ids: Tensor) -> None:
        self._prev_in_contact[env_ids] = False

def reset_box_pose_and_mass(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    x_range: tuple[float, float] = (0.5, 1.5),
    y_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    mass_range: tuple[float, float] = (1.0, 20.0),
    z_clearance: float = 0.3,
    box_name: str = "box",
    geom_name: str = "box_geom",
    robot_name: str = "robot",
) -> None:
    """Reset box pose, clearance above ground, and random mass."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    box = env.scene[box_name]
    assert not box.is_fixed_base, "box must have a free joint"

    n = len(env_ids)
    device = env.device

    robot = env.scene[robot_name]
    q_adr = robot.indexing.free_joint_q_adr
    robot_qpos = robot.data.data.qpos[env_ids][:, q_adr]
    robot_pos_w = robot_qpos[:, :3]
    robot_quat_w = robot_qpos[:, 3:7]
    robot_yaw_quat = yaw_quat(robot_quat_w)

    xs = sample_uniform(
        torch.tensor([x_range[0]], device=device),
        torch.tensor([x_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)
    ys = sample_uniform(
        torch.tensor([y_range[0]], device=device),
        torch.tensor([y_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)
    yaws = sample_uniform(
        torch.tensor([yaw_range[0]], device=device),
        torch.tensor([yaw_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)

    local_geom_ids, _ = box.find_geoms(geom_name, preserve_order=True)
    gid = int(box.indexing.geom_ids[local_geom_ids[0]])
    half_h = env.sim.model.geom_size[env_ids, gid, 2]

    zeros = torch.zeros(n, device=device)
    local_xy = torch.stack([xs, ys, zeros], dim=-1)
    world_xy_offset = quat_apply(robot_yaw_quat, local_xy)

    origins = env.scene.env_origins[env_ids]
    push_here = _push_mask(env)[env_ids]

    WALK_ONLY_OFFSET = 100.0
    push_x = robot_pos_w[:, 0] + world_xy_offset[:, 0]
    push_y = robot_pos_w[:, 1] + world_xy_offset[:, 1]
    parked_x = origins[:, 0] + WALK_ONLY_OFFSET
    parked_y = origins[:, 1]
    positions = torch.stack([
        torch.where(push_here, push_x, parked_x),
        torch.where(push_here, push_y, parked_y),
        origins[:, 2] + half_h + z_clearance,
    ], dim=-1)

    yaw_delta = quat_from_euler_xyz(zeros, zeros, yaws)
    quats = quat_mul(robot_yaw_quat, yaw_delta)

    root_state = torch.cat([
        positions,
        quats,
        torch.zeros(n, 3, device=device),
        torch.zeros(n, 3, device=device),
    ], dim=-1)
    box.write_root_state_to_sim(root_state, env_ids=env_ids)

    masses = sample_uniform(
        torch.tensor([mass_range[0]], device=device),
        torch.tensor([mass_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)
    local_body_ids, _ = box.find_bodies("box", preserve_order=True)
    bid = int(box.indexing.body_ids[local_body_ids[0]])
    env.sim.model.body_mass[env_ids, bid] = masses

_HEIGHT_BUCKETS: tuple[float, ...] = (0.35, 0.475, 0.60, 0.725, 0.85)

_WIDTH_RANGE: tuple[float, float] = (0.10, 0.30)
_DEPTH_RANGE: tuple[float, float] = (0.10, 0.30)

def randomize_box_size_discrete(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    buckets: tuple[float, ...] = _HEIGHT_BUCKETS,
    width_range: tuple[float, float] = _WIDTH_RANGE,
    depth_range: tuple[float, float] = _DEPTH_RANGE,
    box_name: str = "box",
    geom_name: str = "box_geom",
) -> None:
    """Randomise box geom_size: discrete height buckets + continuous width/depth."""
    from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int)

    box = env.scene[box_name]
    local_geom_ids, _ = box.find_geoms(geom_name, preserve_order=True)
    gid = int(box.indexing.geom_ids[local_geom_ids[0]])

    n = len(env_ids)
    device = env.device

    bucket_t = torch.tensor(buckets, device=device, dtype=torch.float32)
    h_indices = torch.randint(0, len(buckets), (n,), device=device)
    half_heights = bucket_t[h_indices]

    half_widths = sample_uniform(
        torch.tensor([width_range[0]], device=device),
        torch.tensor([width_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)
    half_depths = sample_uniform(
        torch.tensor([depth_range[0]], device=device),
        torch.tensor([depth_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)

    env.sim.model.geom_size[env_ids, gid, 0] = half_widths
    env.sim.model.geom_size[env_ids, gid, 1] = half_depths
    env.sim.model.geom_size[env_ids, gid, 2] = half_heights

    asset_cfg = SceneEntityCfg(box_name, geom_names=(geom_name,))
    asset_cfg.resolve(env.scene)
    _recompute_geom_bounds(env, env_ids, asset_cfg)

def chest_point_cloud_noisy(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    max_range: float = 2.5,
    clamp_no_hit: bool = True,
    noise_std: float = 0.01,
    dropout_prob: float = 0.05,
) -> Tensor:
    """Like :func:`chest_point_cloud` but with additive noise and random dropout."""
    clean = chest_point_cloud(env, sensor_name, max_range, clamp_no_hit)
    B = clean.shape[0]
    N3 = clean.shape[1]
    N = N3 // 3

    pts = clean.reshape(B, N, 3)

    if noise_std > 0.0:
        pts = pts + torch.randn_like(pts) * noise_std

    if dropout_prob > 0.0:
        mask = torch.rand(B, N, 1, device=pts.device, dtype=pts.dtype) < dropout_prob
        sentinel = torch.tensor(
            [0.0, 0.0, max_range], device=pts.device, dtype=pts.dtype
        )
        pts = torch.where(mask, sentinel.view(1, 1, 3).expand_as(pts), pts)

    return pts.reshape(B, N3)

def robot_far_from_box(
    env: "ManagerBasedRlEnv",
    max_distance: float = 2.0,
    box_name: str = "box",
    robot_name: str = "robot",
) -> Tensor:
    """Terminate episode when robot is too far from the box (XY plane)."""
    box_pos_w = env.scene[box_name].data.root_link_pos_w
    base_pos_w = env.scene[robot_name].data.root_link_pos_w
    delta_xy = (box_pos_w[:, :2] - base_pos_w[:, :2])
    dist = delta_xy.norm(dim=-1)
    return (dist > max_distance) & _push_mask(env)

def box_toppled(
    env: "ManagerBasedRlEnv",
    max_tilt_rad: float = 1.0,
    box_name: str = "box",
) -> Tensor:
    """Terminate when the box has toppled over (tilted beyond threshold)."""
    box_quat = env.scene[box_name].data.root_link_quat_w
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=box_quat.dtype)
    box_up = quat_apply(box_quat, world_up.expand(box_quat.shape[0], 3))
    cos_angle = box_up[:, 2]
    angle = torch.acos(cos_angle.clamp(-1.0, 1.0))
    return (angle > max_tilt_rad) & _push_mask(env)

def box_pitched(
    env: "ManagerBasedRlEnv",
    max_pitch_rad: float = 0.1745,
    box_name: str = "box",
) -> Tensor:
    """Terminate when the box pitch exceeds ``max_pitch_rad``."""
    box_quat = env.scene[box_name].data.root_link_quat_w
    w, x, y, z = box_quat[:, 0], box_quat[:, 1], box_quat[:, 2], box_quat[:, 3]
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(sinp.clamp(-1.0, 1.0))
    return (pitch.abs() > max_pitch_rad) & _push_mask(env)

def arm_joint_deviation(
    env: "ManagerBasedRlEnv",
    max_deviation_rad: float = 1.2,
    robot_name: str = "robot",
) -> Tensor:
    """Terminate when any arm or neck joint deviates too far from default."""
    robot = env.scene[robot_name]
    q = robot.data.joint_pos
    q_default = robot.data.default_joint_pos
    if not hasattr(env, "_arm_neck_joint_ids"):
        ids, _ = robot.find_joints(
            [
                r".*SHOULDER.*", r".*ELBOW.*", r".*WRIST.*", r".*HEAD.*",
            ],
            preserve_order=False,
        )
        env._arm_neck_joint_ids = ids
    ids = env._arm_neck_joint_ids
    deviation = (q[:, ids] - q_default[:, ids]).abs()
    return deviation.amax(dim=-1) > max_deviation_rad

def box_velocity_stall(
    env: "ManagerBasedRlEnv",
    min_speed_fraction: float = 0.5,
    grace_period_s: float = 5.0,
    activate_after_step: int = 2500,
    box_name: str = "box",
    command_name: str = "twist",
) -> Tensor:
    """Terminate when the box fails to reach a minimum fraction of commanded speed."""
    B      = env.num_envs
    device = env.device

    if env.common_step_counter < activate_after_step:
        return torch.zeros(B, dtype=torch.bool, device=device)

    if not hasattr(env, "_box_stall_time"):
        env._box_stall_time = torch.zeros(B, device=device, dtype=torch.float32)

    just_reset = (env.episode_length_buf == 1)
    env._box_stall_time[just_reset] = 0.0

    cmd = env.command_manager.get_command(command_name)
    if cmd is None:
        return torch.zeros(B, dtype=torch.bool, device=device)
    cmd_speed = cmd[:, :2].norm(dim=-1)

    box_vel = env.scene[box_name].data.root_link_lin_vel_w[:, :2]
    box_speed = box_vel.norm(dim=-1)

    is_cmd_active = cmd_speed > 0.05
    box_is_slow   = box_speed < (min_speed_fraction * cmd_speed)
    stalling      = is_cmd_active & box_is_slow

    env._box_stall_time = torch.where(
        stalling,
        env._box_stall_time + env.step_dt,
        torch.zeros_like(env._box_stall_time),
    )

    timed_out = env._box_stall_time > grace_period_s
    return timed_out & _push_mask(env)

def push_box_mass_size_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor,
    mass_size_stages: list[dict],
) -> Tensor:
    """Progressively widen box mass, size, and ground-friction ranges."""
    del env_ids
    for stage in mass_size_stages:
        if env.common_step_counter > stage["step"]:
            if "mass_range" in stage:
                env._push_box_mass_range = tuple(stage["mass_range"])
            if "height_buckets" in stage:
                env._push_box_height_buckets = tuple(stage["height_buckets"])
            if "friction_range" in stage:
                env._push_box_friction_range = tuple(stage["friction_range"])

    mass_range = getattr(env, "_push_box_mass_range", (1.0, 20.0))
    buckets = getattr(env, "_push_box_height_buckets", _HEIGHT_BUCKETS)
    friction_range = getattr(env, "_push_box_friction_range", (0.8, 0.8))
    return {
        "mass_min": torch.tensor(mass_range[0]),
        "mass_max": torch.tensor(mass_range[1]),
        "num_buckets": torch.tensor(float(len(buckets))),
        "friction_min": torch.tensor(friction_range[0]),
        "friction_max": torch.tensor(friction_range[1]),
    }

def randomize_box_friction_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    default_friction_range: tuple[float, float] = (0.8, 0.8),
    box_name: str = "box",
    geom_name: str = "box_geom",
) -> None:
    """Sample box tangential friction from the curriculum range (reset event)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int)

    fr_range = getattr(env, "_push_box_friction_range", default_friction_range)
    box = env.scene[box_name]
    local_geom_ids, _ = box.find_geoms(geom_name, preserve_order=True)
    gid = int(box.indexing.geom_ids[local_geom_ids[0]])

    n = len(env_ids)
    device = env.device
    frictions = sample_uniform(
        torch.tensor([fr_range[0]], device=device),
        torch.tensor([fr_range[1]], device=device),
        (n, 1), device=device,
    ).squeeze(-1)
    env.sim.model.geom_friction[env_ids, gid, 0] = frictions

def reset_box_pose_and_mass_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    x_range: tuple[float, float] = (0.5, 1.5),
    y_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    default_mass_range: tuple[float, float] = (1.0, 20.0),
    z_clearance: float = 0.3,
    box_name: str = "box",
    geom_name: str = "box_geom",
) -> None:
    """Like :func:`reset_box_pose_and_mass` but reads mass range from curriculum."""
    mass_range = getattr(env, "_push_box_mass_range", default_mass_range)
    reset_box_pose_and_mass(
        env, env_ids,
        x_range=x_range,
        y_range=y_range,
        yaw_range=yaw_range,
        mass_range=mass_range,
        z_clearance=z_clearance,
        box_name=box_name,
        geom_name=geom_name,
    )

def randomize_box_size_discrete_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: Tensor | None,
    default_buckets: tuple[float, ...] = _HEIGHT_BUCKETS,
    default_width_range: tuple[float, float] = _WIDTH_RANGE,
    default_depth_range: tuple[float, float] = _DEPTH_RANGE,
    box_name: str = "box",
    geom_name: str = "box_geom",
) -> None:
    """Like :func:`randomize_box_size_discrete` but reads buckets from curriculum."""
    buckets = getattr(env, "_push_box_height_buckets", default_buckets)
    randomize_box_size_discrete(
        env, env_ids,
        buckets=buckets,
        width_range=default_width_range,
        depth_range=default_depth_range,
        box_name=box_name,
        geom_name=geom_name,
    )
