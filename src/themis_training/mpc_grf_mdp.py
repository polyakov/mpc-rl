"""Centroidal locomotion MPC + GRF tracking pipeline for THEMIS training."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from themis_mpc.centroidal_mpc import CentroidalMPC, MPCConfig, MPCInput
from themis_mpc.contact_schedule import make_walking_schedule
from themis_mpc.loco_manip_mpc import LocoManipMPC, LocoManipMPCConfig, LocoManipMPCInput

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_I_BODY: tuple[tuple[float, ...], ...] = (
    ( 6.153, 0.0,   0.338),
    ( 0.0,   6.181, 0.0  ),
    ( 0.338, 0.0,   0.849),
)

def _quat_to_rot(q: Tensor) -> Tensor:
    """Convert unit quaternions (B, 4) [w, x, y, z] to rotation matrices (B, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(q.shape[0], 3, 3)

@dataclass(kw_only=True)
class LocoMPCCommandCfg(CommandTermCfg):
    """Configuration for :class:`LocoMPCCommand`."""

    asset_cfg: SceneEntityCfg = field(
        default_factory=lambda: SceneEntityCfg(
            "robot",
            site_names=("left_foot", "right_foot"),
        )
    )

    mpc_dt: float = 0.07
    mpc_horizon: int = 10
    mass: float = 37.0
    hip_width: float = 0.1

    gait_period: float = 0.9
    duty_factor: float = 0.5

    com_height: float = 1.17

    vel_cmd_name: str = "twist"

    grf_sensor_name: str = "feet_ground_contact"

    run_every_n_steps: int = 5

    tracking_lookahead_frac: float = 0

    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    unconstrained: bool = False

    solver_type: str = "pimpc"

    def build(self, env: "ManagerBasedRlEnv") -> "LocoMPCCommand":
        return LocoMPCCommand(self, env)

class LocoMPCCommand(CommandTerm):
    """Centroidal locomotion MPC runner as a CommandTerm."""

    cfg: LocoMPCCommandCfg

    def __init__(self, cfg: LocoMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        self._asset_cfg = copy.deepcopy(cfg.asset_cfg)
        self._asset_cfg.resolve(env.scene)
        self._robot = env.scene[self._asset_cfg.name]
        self._site_ids: list[int] = self._asset_cfg.site_ids

        _, _resolved_names = self._robot.find_sites(
            cfg.asset_cfg.site_names, preserve_order=False
        )
        self._lf_site_local_idx = next(
            i for i, n in enumerate(_resolved_names) if n == "left_foot"
        )
        self._rf_site_local_idx = next(
            i for i, n in enumerate(_resolved_names) if n == "right_foot"
        )

        self._I_approx = torch.tensor(
            _I_BODY, device=env.device, dtype=torch.float32
        )

        mpc_cfg = MPCConfig(
            N=cfg.mpc_horizon,
            dt=cfg.mpc_dt,
            mass=cfg.mass,
            solver_type=cfg.solver_type,
            unconstrained=cfg.unconstrained,
        )
        self._mpc = CentroidalMPC(mpc_cfg, device=env.device)

        B = env.num_envs
        N = cfg.mpc_horizon
        self._grf_ref: Tensor = torch.zeros(B, 6, device=env.device)
        self._u_prev: Tensor = torch.zeros(B, 12, device=env.device)
        self._step_count: int = 0

        self._com_traj:  Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vel_traj:  Tensor = torch.zeros(B, N, 3, device=env.device)
        self._k_traj:    Tensor = torch.zeros(B, N, 3, device=env.device)
        self._kdot_traj: Tensor = torch.zeros(B, N, 3, device=env.device)
        self._traj_step: int = 0

        self._com_mpc_target:      Tensor = torch.zeros(B, 3, device=env.device)
        self._com_vel_mpc_target:  Tensor = torch.zeros(B, 3, device=env.device)
        self._k_mpc_target:        Tensor = torch.zeros(B, 3, device=env.device)
        self._k_dot_mpc_target:    Tensor = torch.zeros(B, 3, device=env.device)

        self._vis_com:      Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_ang_mom:  Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_r_lf:     Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_r_rf:     Tensor = torch.zeros(B, N, 3, device=env.device)
        self._vis_sigma_lf: Tensor = torch.zeros(B, N, device=env.device)
        self._vis_sigma_rf: Tensor = torch.zeros(B, N, device=env.device)

        self._lf_landing_target: Tensor = torch.zeros(B, 3, device=env.device)
        self._rf_landing_target: Tensor = torch.zeros(B, 3, device=env.device)
        self._lf_landing_valid:  Tensor = torch.zeros(B, dtype=torch.bool, device=env.device)
        self._rf_landing_valid:  Tensor = torch.zeros(B, dtype=torch.bool, device=env.device)

    @property
    def command(self) -> Tensor:
        """Optimal first-step foot forces [B, 6]: [fLF(3), fRF(3)]."""
        return self._grf_ref

    def _interpolate_traj_refs(self) -> None:
        """Interpolate current references from stored MPC trajectories."""
        N = self.cfg.mpc_horizon
        elapsed = self._traj_step * self._env.step_dt
        t_frac_slide = elapsed / self.cfg.mpc_dt
        base = self.cfg.tracking_lookahead_frac * (N - 1)
        t_frac = base + t_frac_slide

        idx = min(max(int(t_frac), 0), N - 2)
        alpha = min(max(t_frac - idx, 0.0), 1.0)

        self._com_mpc_target     = (1 - alpha) * self._com_traj[:, idx]  + alpha * self._com_traj[:, idx + 1]
        self._com_vel_mpc_target = (1 - alpha) * self._vel_traj[:, idx]  + alpha * self._vel_traj[:, idx + 1]
        self._k_mpc_target       = (1 - alpha) * self._k_traj[:, idx]    + alpha * self._k_traj[:, idx + 1]
        self._k_dot_mpc_target   = (1 - alpha) * self._kdot_traj[:, idx] + alpha * self._kdot_traj[:, idx + 1]

    def _update_command(self) -> None:
        """Solve the centroidal MPC and update the GRF reference."""
        self._step_count += 1
        self._traj_step += 1
        self._interpolate_traj_refs()

        if self._step_count % self.cfg.run_every_n_steps != 0:
            return

        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        dt = cfg.mpc_dt
        device = self.device
        robot = self._robot

        c  = robot.data.root_link_pos_w
        lv = robot.data.root_link_lin_vel_w
        av = robot.data.root_link_ang_vel_w
        l  = lv * cfg.mass
        k  = av @ self._I_approx
        x0 = torch.cat([c, l, k], dim=-1)

        site_pos = robot.data.site_pos_w[:, self._site_ids, :]
        r_lf = site_pos[:, self._lf_site_local_idx, :]
        r_rf = site_pos[:, self._rf_site_local_idx, :]

        vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
        if vel_cmd is not None:
            vx_body = vel_cmd[:, 0]
            vy_body = vel_cmd[:, 1]
            wz      = vel_cmd[:, 2]
        else:
            vx_body = torch.zeros(B, device=device)
            vy_body = torch.zeros(B, device=device)
            wz      = torch.zeros(B, device=device)

        quat_w = robot.data.root_link_quat_w
        q_w, q_x, q_y, q_z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
        yaw   = torch.atan2(2.0 * (q_w * q_z + q_x * q_y),
                             1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        cos_y = yaw.cos()
        sin_y = yaw.sin()
        vx    = cos_y * vx_body - sin_y * vy_body
        vy    = sin_y * vx_body + cos_y * vy_body

        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

        k_steps = torch.arange(N + 1, device=device, dtype=torch.float32)
        yaw_k   = yaw.unsqueeze(1) + wz.unsqueeze(1) * k_steps * dt
        cos_k   = yaw_k.cos()
        sin_k   = yaw_k.sin()

        vx_w_k = cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1)
        vy_w_k = sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1)

        x_ref = x0.unsqueeze(1).expand(B, N + 1, -1).clone()
        x_ref[:, 0, 0] = c[:, 0]
        x_ref[:, 0, 1] = c[:, 1]
        x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w_k[:, :-1] * dt, dim=1)
        x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w_k[:, :-1] * dt, dim=1)
        z_init = robot.data.default_root_state[:, 2:3]
        x_ref[:, :, 2] = z_init

        x_ref[:, :, 3] = vx_w_k * cfg.mass
        x_ref[:, :, 4] = vy_w_k * cfg.mass
        x_ref[:, :, 5] = 0.0

        I_zz = float(self._I_approx[2, 2])
        x_ref[:, :, 6] = 0.0
        x_ref[:, :, 7] = 0.0
        x_ref[:, :, 8] = (I_zz * wz).unsqueeze(1)

        gait_phase = getattr(
            self._env, "_gait_phase", torch.zeros(B, device=device)
        )
        v_cmd_3d = torch.zeros(B, 3, device=device)
        v_cmd_3d[:, 0] = vx
        v_cmd_3d[:, 1] = vy
        schedule = make_walking_schedule(
            B=B,
            N=N,
            r_LF=r_lf,
            r_RF=r_rf,
            gait_phase=gait_phase,
            period=cfg.gait_period,
            dt=dt,
            duty_factor=cfg.duty_factor,
            com_pos=c,
            v_cmd=v_cmd_3d,
            yaw=yaw,
            yaw_rate=wz,
            hip_width=cfg.hip_width,
            R_LF_rot=R_lf,
            R_RF_rot=R_rf,
            device=device,
        )

        u_ref = torch.zeros(B, N, 12, device=device)

        mpc_in = MPCInput(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            c_bar=None,
        )
        with torch.no_grad():
            mpc_out = self._mpc.solve(mpc_in)

        self._grf_ref = torch.cat([
            mpc_out.u_star[:, 0:3],
            mpc_out.u_star[:, 6:9],
        ], dim=-1)

        self._u_prev = mpc_out.u_star.detach()

        x_pred = mpc_out.x_pred.detach()
        self._com_traj = x_pred[:, :, 0:3]
        l_traj = x_pred[:, :, 3:6]
        self._vel_traj = l_traj / cfg.mass
        k_pred = x_pred[:, :, 6:9]
        self._k_traj = k_pred
        self._kdot_traj = torch.zeros_like(k_pred)
        if N >= 2:
            self._kdot_traj[:, :-1] = (k_pred[:, 1:] - k_pred[:, :-1]) / cfg.mpc_dt
            self._kdot_traj[:, -1] = self._kdot_traj[:, -2]

        self._traj_step = 0
        self._interpolate_traj_refs()

        self._vis_com      = self._com_traj
        self._vis_ang_mom  = self._k_traj
        self._vis_r_lf     = schedule.r_LF.detach()
        self._vis_r_rf     = schedule.r_RF.detach()
        self._vis_sigma_lf = schedule.sigma[:, :, 0].detach()
        self._vis_sigma_rf = schedule.sigma[:, :, 1].detach()

        sigma_lf = schedule.sigma[:, :, 0]
        sigma_rf = schedule.sigma[:, :, 1]

        lf_prev = torch.cat([sigma_lf[:, :1], sigma_lf[:, :-1]], dim=1)
        rf_prev = torch.cat([sigma_rf[:, :1], sigma_rf[:, :-1]], dim=1)
        lf_td = (sigma_lf > 0.5) & (lf_prev < 0.5)
        rf_td = (sigma_rf > 0.5) & (rf_prev < 0.5)

        self._lf_landing_valid = lf_td.any(dim=1)
        self._rf_landing_valid = rf_td.any(dim=1)
        lf_idx = lf_td.float().argmax(dim=1)
        rf_idx = rf_td.float().argmax(dim=1)

        batch_idx = torch.arange(B, device=device)
        self._lf_landing_target = torch.where(
            self._lf_landing_valid.unsqueeze(-1),
            schedule.r_LF[batch_idx, lf_idx],
            r_lf,
        )
        self._rf_landing_target = torch.where(
            self._rf_landing_valid.unsqueeze(-1),
            schedule.r_RF[batch_idx, rf_idx],
            r_rf,
        )

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Draw MPC horizon visualizations."""
        import numpy as np

        B = self.num_envs
        N = self.cfg.mpc_horizon
        env_indices = visualizer.get_env_indices(B)
        if not env_indices:
            return

        com_np  = self._vis_com.cpu().numpy()
        k_np    = self._vis_ang_mom.cpu().numpy()
        r_lf_np = self._vis_r_lf.cpu().numpy()
        r_rf_np = self._vis_r_rf.cpu().numpy()
        s_lf_np = self._vis_sigma_lf.cpu().numpy()
        s_rf_np = self._vis_sigma_rf.cpu().numpy()

        _EPS      = 1e-4
        _S_MIN    = 0.08
        _ALPHA    = 0.05
        _S_PERP   = 0.12
        _GAMMA    = 0.08
        _K_NODES  = [0, N // 2, N - 1]
        _K_ALPHAS = [0.75, 0.45, 0.15]

        for b in env_indices:
            for k in range(N):
                alpha = 0.85 * (1.0 - k / N) + 0.05

                visualizer.add_sphere(
                    center=com_np[b, k],
                    radius=0.03,
                    color=(0.85, 0.15, 0.15, alpha),
                    label=f"mpc_com_{b}_{k}",
                )
                if k < N - 1:
                    visualizer.add_cylinder(
                        start=com_np[b, k],
                        end=com_np[b, k + 1],
                        radius=0.012,
                        color=(0.85, 0.15, 0.15, alpha * 0.6),
                        label=f"mpc_com_line_{b}_{k}",
                    )

            for foot, r_np, s_np, rgb in (
                ("lf", r_lf_np, s_lf_np, (0.2, 0.5, 1.0)),
                ("rf", r_rf_np, s_rf_np, (1.0, 0.45, 0.15)),
            ):
                if s_np[b, 0] > 0.5:
                    pos = r_np[b, 0].copy()
                    pos[2] = max(pos[2], 0.01)
                    visualizer.add_sphere(
                        center=pos,
                        radius=0.045,
                        color=(*rgb, 0.9),
                        label=f"mpc_{foot}_cur_{b}",
                    )
                for k in range(1, N):
                    if s_np[b, k] > 0.5 and s_np[b, k - 1] < 0.5:
                        pos = r_np[b, k].copy()
                        pos[2] = max(pos[2], 0.01)
                        visualizer.add_sphere(
                            center=pos,
                            radius=0.045,
                            color=(*rgb, 0.5),
                            label=f"mpc_{foot}_td{k}_{b}",
                        )

            for ki, a_glyph in zip(_K_NODES, _K_ALPHAS):
                c_i = com_np[b, ki]
                k_i = k_np[b, ki]
                norm_k = float(np.linalg.norm(k_i))

                if norm_k > _EPS:
                    e1 = k_i / norm_k
                    ref = (np.array([0.0, 0.0, 1.0])
                           if abs(e1 @ np.array([0.0, 0.0, 1.0])) < 0.9
                           else np.array([1.0, 0.0, 0.0]))
                    e2_raw = ref - (e1 @ ref) * e1
                    e2 = e2_raw / np.linalg.norm(e2_raw)
                    e3 = np.cross(e1, e2)
                else:
                    e1 = np.array([1.0, 0.0, 0.0])
                    e2 = np.array([0.0, 1.0, 0.0])
                    e3 = np.array([0.0, 0.0, 1.0])

                R = np.stack([e1, e2, e3], axis=1)

                s1 = _S_MIN + _ALPHA * norm_k
                size = np.array([s1, _S_PERP, _S_PERP], dtype=np.float64)

                visualizer.add_ellipsoid(
                    center=c_i,
                    size=size,
                    mat=R,
                    color=(0.7, 0.3, 1.0, a_glyph),
                    label=f"mpc_k_ellipsoid_{b}_{ki}",
                )

                if norm_k > _EPS:
                    visualizer.add_arrow(
                        start=c_i,
                        end=c_i + _GAMMA * k_i,
                        color=(0.7, 0.3, 1.0, min(a_glyph + 0.2, 1.0)),
                        width=0.012,
                        label=f"mpc_k_arrow_{b}_{ki}",
                    )

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Tensor) -> None:
        """Reset MPC warm start for terminated / reset environments."""
        self._grf_ref[env_ids]             = 0.0
        self._u_prev[env_ids]              = 0.0
        self._com_traj[env_ids]            = 0.0
        self._vel_traj[env_ids]            = 0.0
        self._k_traj[env_ids]              = 0.0
        self._kdot_traj[env_ids]           = 0.0
        self._com_mpc_target[env_ids]      = 0.0
        self._com_vel_mpc_target[env_ids]  = 0.0
        self._k_mpc_target[env_ids]        = 0.0
        self._k_dot_mpc_target[env_ids]    = 0.0
        self._vis_ang_mom[env_ids]         = 0.0
        self._mpc.reset(env_ids)

def mpc_grf_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    grf_sensor_name: str = "feet_ground_contact",
) -> Tensor:
    """GRF tracking error vs. MPC reference — use with weight = 0.0."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    grf_ref: Tensor = term._grf_ref

    sensor: ContactSensor = env.scene[grf_sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force = sensor.data.force
    grf_actual = force.reshape(env.num_envs, 6)

    rms_err = (grf_actual - grf_ref).pow(2).mean(dim=-1).sqrt()
    return -rms_err

def mpc_com_ref(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
) -> Tensor:
    """Expose the MPC CoM-position target to the policy."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, 3, device=env.device)
    c_cur = term._robot.data.root_link_pos_w
    return term._com_mpc_target - c_cur

def mpc_ang_mom_ref(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
) -> Tensor:
    """Expose the MPC angular-momentum target ``k_t^mpc`` to the policy."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return term._k_mpc_target

class MpcAngMomTracking:
    """Soft MPC-guided angular-momentum regulation reward."""

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv") -> None:  # noqa: F821
        self._env = env
        self._k_prev: Tensor = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str = "loco_mpc",
        w_k: float = 1.0,
        w_kdot: float = 0.01,
        q_k: tuple[float, float, float] = (1.0, 1.0, 0.5),
        q_kdot: tuple[float, float, float] = (0.1, 0.1, 0.05),
        lookahead_fracs: tuple[float, ...] | None = None,
        lookahead_weights: tuple[float, ...] | None = None,
    ) -> Tensor:
        """Angular-momentum tracking loss (≤ 0)."""
        term = env.command_manager.get_term(command_name)
        if term is None or not isinstance(term, LocoMPCCommand):
            return torch.zeros(env.num_envs, device=env.device)

        av = term._robot.data.root_link_ang_vel_w
        k_t = av @ term._I_approx

        dt = env.step_dt
        k_dot_t = (k_t - self._k_prev) / dt
        self._k_prev = k_t.detach()

        Q_k  = torch.tensor(q_k,    device=env.device, dtype=torch.float32)
        Q_dk = torch.tensor(q_kdot, device=env.device, dtype=torch.float32)

        if lookahead_fracs is None:
            k_mpc     = term._k_mpc_target
            k_dot_mpc = term._k_dot_mpc_target
            ek  = k_t   - k_mpc
            edk = k_dot_t - k_dot_mpc
            loss_k    = (ek.pow(2)  * Q_k ).sum(dim=-1)
            loss_kdot = (edk.pow(2) * Q_dk).sum(dim=-1)
            return -(w_k * loss_k + w_kdot * loss_kdot)

        N = term.cfg.mpc_horizon
        k_traj    = term._k_traj
        kdot_traj = term._kdot_traj

        if lookahead_weights is None:
            n = len(lookahead_fracs)
            lookahead_weights = tuple(1.0 / n for _ in range(n))
        if len(lookahead_weights) != len(lookahead_fracs):
            raise ValueError(
                "lookahead_weights must match lookahead_fracs length "
                f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
            )

        total = torch.zeros(env.num_envs, device=env.device)
        for frac, weight in zip(lookahead_fracs, lookahead_weights):
            t = frac * (N - 1)
            idx = min(max(int(t), 0), N - 2)
            alpha = min(max(t - idx, 0.0), 1.0)
            k_ref  = (1 - alpha) * k_traj[:, idx]    + alpha * k_traj[:, idx + 1]
            kd_ref = (1 - alpha) * kdot_traj[:, idx] + alpha * kdot_traj[:, idx + 1]
            ek  = k_t   - k_ref
            edk = k_dot_t - kd_ref
            loss_k    = (ek.pow(2)  * Q_k ).sum(dim=-1)
            loss_kdot = (edk.pow(2) * Q_dk).sum(dim=-1)
            total = total + weight * (w_k * loss_k + w_kdot * loss_kdot)

        return -total

    def reset(self, env_ids: Tensor) -> None:
        """Clear finite-difference history for reset environments."""
        self._k_prev[env_ids] = 0.0

def mpc_com_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    w_pos: float = 1.0,
    w_vel: float = 0.5,
    lookahead_fracs: tuple[float, ...] | None = None,
    lookahead_weights: tuple[float, ...] | None = None,
) -> Tensor:
    """Reward for tracking the MPC-predicted CoM position and velocity."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    c_cur = term._robot.data.root_link_pos_w
    v_cur = term._robot.data.root_link_lin_vel_w

    if lookahead_fracs is None:
        e_pos = c_cur - term._com_mpc_target
        e_vel = v_cur - term._com_vel_mpc_target
        return torch.exp(
            -w_pos * e_pos.pow(2).sum(dim=-1)
            -w_vel * e_vel.pow(2).sum(dim=-1)
        )

    N = term.cfg.mpc_horizon
    com_traj = term._com_traj
    vel_traj = term._vel_traj

    if lookahead_weights is None:
        k = len(lookahead_fracs)
        lookahead_weights = tuple(1.0 / k for _ in range(k))
    if len(lookahead_weights) != len(lookahead_fracs):
        raise ValueError(
            "lookahead_weights must match lookahead_fracs length "
            f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
        )

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        c_ref = (1 - alpha) * com_traj[:, idx] + alpha * com_traj[:, idx + 1]
        v_ref = (1 - alpha) * vel_traj[:, idx] + alpha * vel_traj[:, idx + 1]
        e_pos = c_cur - c_ref
        e_vel = v_cur - v_ref
        total = total + weight * (
            w_pos * e_pos.pow(2).sum(dim=-1)
            + w_vel * e_vel.pow(2).sum(dim=-1)
        )

    return torch.exp(-total)

class TrackingMetricsVelocityCommand(UniformVelocityCommand):
    """:class:`UniformVelocityCommand` augmented with exp-kernel tracking"""

    cfg: "TrackingMetricsVelocityCommandCfg"

    def __init__(
        self,
        cfg: "TrackingMetricsVelocityCommandCfg",
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(cfg, env)
        self.metrics["tracking_lin_score"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["tracking_ang_score"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _update_metrics(self) -> None:
        super()._update_metrics()

        actual_lin = self.robot.data.root_link_lin_vel_b
        xy_err_sq = torch.sum(
            torch.square(self.vel_command_b[:, :2] - actual_lin[:, :2]), dim=1
        )
        z_err_sq = torch.square(actual_lin[:, 2])
        lin_score = torch.exp(
            -(xy_err_sq + z_err_sq) / (self.cfg.metric_std_lin ** 2)
        )
        self.metrics["tracking_lin_score"] += lin_score

        actual_ang = self.robot.data.root_link_ang_vel_b
        yaw_err_sq = torch.square(self.vel_command_b[:, 2] - actual_ang[:, 2])
        xy_ang_sq = torch.sum(torch.square(actual_ang[:, :2]), dim=1)
        ang_score = torch.exp(
            -(yaw_err_sq + xy_ang_sq) / (self.cfg.metric_std_ang ** 2)
        )
        self.metrics["tracking_ang_score"] += ang_score

    def reset(self, env_ids: Tensor | slice | None) -> dict[str, float]:
        """Normalise the accumulated tracking-score sums by ``N_max`` so the"""
        if isinstance(env_ids, torch.Tensor):
            n_max = max(int(self._env.max_episode_length), 1)
            self.metrics["tracking_lin_score"][env_ids] = (
                self.metrics["tracking_lin_score"][env_ids] / n_max
            )
            self.metrics["tracking_ang_score"][env_ids] = (
                self.metrics["tracking_ang_score"][env_ids] / n_max
            )
        return super().reset(env_ids)

@dataclass(kw_only=True)
class TrackingMetricsVelocityCommandCfg(UniformVelocityCommandCfg):
    """Drop-in replacement for :class:`UniformVelocityCommandCfg` that emits"""

    metric_std_lin: float = 0.5
    metric_std_ang: float = 0.7071067811865476

    def build(self, env: "ManagerBasedRlEnv") -> TrackingMetricsVelocityCommand:
        return TrackingMetricsVelocityCommand(self, env)

    @classmethod
    def from_uniform(
        cls,
        base: UniformVelocityCommandCfg,
        **overrides,
    ) -> "TrackingMetricsVelocityCommandCfg":
        """Construct from an existing :class:`UniformVelocityCommandCfg`,"""
        import dataclasses
        fields = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(base)
        }
        fields.update(overrides)
        return cls(**fields)

def mpc_com_vel_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    w_vel: float = 1.0,
    lookahead_fracs: tuple[float, ...] | None = (0.0, 0.5, 1.0),
    lookahead_weights: tuple[float, ...] | None = None,
) -> Tensor:
    """Reward for tracking the MPC-predicted CoM **velocity** trajectory."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    v_cur = term._robot.data.root_link_lin_vel_w

    if lookahead_fracs is None:
        e_vel = v_cur - term._com_vel_mpc_target
        return torch.exp(-w_vel * e_vel.pow(2).sum(dim=-1))

    N = term.cfg.mpc_horizon
    vel_traj = term._vel_traj

    if lookahead_weights is None:
        k = len(lookahead_fracs)
        lookahead_weights = tuple(1.0 / k for _ in range(k))
    if len(lookahead_weights) != len(lookahead_fracs):
        raise ValueError(
            "lookahead_weights must match lookahead_fracs length "
            f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
        )

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        v_ref = (1 - alpha) * vel_traj[:, idx] + alpha * vel_traj[:, idx + 1]
        e_vel = v_cur - v_ref
        total = total + weight * w_vel * e_vel.pow(2).sum(dim=-1)

    return torch.exp(-total)

def mpc_ang_vel_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    w_ang: float = 2.0,
    lookahead_fracs: tuple[float, ...] | None = (0.0, 0.5, 1.0),
    lookahead_weights: tuple[float, ...] | None = None,
) -> Tensor:
    """Reward for tracking the MPC-predicted body **angular velocity** trajectory."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    omega_cur = term._robot.data.root_link_ang_vel_w

    I_inv = torch.linalg.inv(term._I_approx)

    if lookahead_fracs is None:
        omega_mpc = term._k_mpc_target @ I_inv
        e = omega_cur - omega_mpc
        return torch.exp(-w_ang * e.pow(2).sum(dim=-1))

    N = term.cfg.mpc_horizon
    k_traj = term._k_traj

    if lookahead_weights is None:
        k_n = len(lookahead_fracs)
        lookahead_weights = tuple(1.0 / k_n for _ in range(k_n))
    if len(lookahead_weights) != len(lookahead_fracs):
        raise ValueError(
            "lookahead_weights must match lookahead_fracs length "
            f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
        )

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        k_ref = (1 - alpha) * k_traj[:, idx] + alpha * k_traj[:, idx + 1]
        omega_ref = k_ref @ I_inv
        e = omega_cur - omega_ref
        total = total + weight * w_ang * e.pow(2).sum(dim=-1)

    return torch.exp(-total)

class MpcComCLFTracking:
    """CLF-based reward for tracking the MPC-predicted CoM evolution."""

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv") -> None:  # noqa: F821
        import numpy as np
        from scipy.linalg import solve_continuous_are

        params = dict(cfg.params)
        q_pos = float(params.get("q_pos", 5.0))
        q_vel = float(params.get("q_vel", 1.0))
        r_in  = float(params.get("r_input", 0.1))
        lam   = float(params.get("lambda_decay", 4.0))
        eta_max     = float(params.get("eta_max", 0.10))
        eta_dot_max = float(params.get("eta_dot_max", 0.50))

        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        R = np.diag([r_in, r_in, r_in])
        P_np = solve_continuous_are(A, B, Q, R)

        P_np = 0.5 * (P_np + P_np.T)
        mu_max = float(np.linalg.eigvalsh(P_np).max())
        p_norm = float(np.linalg.norm(P_np, ord=2))

        self._env = env
        self._P: Tensor = torch.tensor(
            P_np, device=env.device, dtype=torch.float32
        )
        self._lambda: float = lam
        self._sigma_v: float = mu_max * (eta_max ** 2) + 1e-6
        self._sigma_v_dot: float = (
            2.0 * p_norm * eta_max * eta_dot_max
            + lam * mu_max * (eta_max ** 2)
            + 1e-6
        )

        self._V_prev: Tensor = torch.zeros(env.num_envs, device=env.device)
        self._has_prev: Tensor = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    def _lyapunov(self, eta: Tensor) -> Tensor:
        """Compute :math:`V(\\eta) = \\eta^\\top P \\eta` per env (B,)."""
        return ((eta @ self._P) * eta).sum(dim=-1)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str = "loco_mpc",
        w_v: float = 1.0,
        w_v_dot: float = 0.5,
        lambda_decay: float = 4.0,    # noqa: ARG002
        q_pos: float = 5.0,           # noqa: ARG002
        q_vel: float = 1.0,           # noqa: ARG002
        r_input: float = 0.1,         # noqa: ARG002
        eta_max: float = 0.10,        # noqa: ARG002
        eta_dot_max: float = 0.50,    # noqa: ARG002
    ) -> Tensor:
        term = env.command_manager.get_term(command_name)
        if term is None or not isinstance(term, LocoMPCCommand):
            return torch.zeros(env.num_envs, device=env.device)

        c_cur = term._robot.data.root_link_pos_w
        v_cur = term._robot.data.root_link_lin_vel_w
        c_ref = term._com_mpc_target
        v_ref = term._com_vel_mpc_target

        eta = torch.cat([c_ref - c_cur, v_ref - v_cur], dim=-1)
        V = self._lyapunov(eta)

        r_val = torch.exp(-V / self._sigma_v)

        dt = env.step_dt
        V_dot = (V - self._V_prev) / dt
        decay_term = (V_dot + self._lambda * V) / self._sigma_v_dot
        decay_term = torch.clamp(decay_term, min=0.0, max=1.0)
        decay_term = torch.where(
            self._has_prev, decay_term, torch.zeros_like(decay_term)
        )

        self._V_prev = V.detach()
        self._has_prev[:] = True

        return w_v * r_val - w_v_dot * decay_term

    def reset(self, env_ids: Tensor) -> None:
        """Clear FD history for reset envs so the decay term resumes cleanly."""
        self._V_prev[env_ids] = 0.0
        self._has_prev[env_ids] = False

def foot_flat_orientation(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
    sigma: float = 0.15,
) -> Tensor:
    """Reward flat foot orientation — active in both swing and stance."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.site_quat_w[:, asset_cfg.site_ids, :]
    tilt = 2.0 * (q[:, :, 1].pow(2) + q[:, :, 2].pow(2))
    return torch.exp(-tilt / sigma).mean(dim=-1)

def mpc_foot_placement_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    sigma: float = 0.2,
    command_threshold: float = 0.1,
) -> Tensor:
    """Reward swing feet for tracking MPC-predicted landing positions."""
    import math as _math

    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    B = env.num_envs
    device = env.device

    gait_phase = getattr(env, "_gait_phase", torch.zeros(B, device=device))
    threshold = _math.cos(_math.pi * term.cfg.duty_factor)
    lf_in_swing = (gait_phase + _math.pi).sin() >= threshold
    rf_in_swing = gait_phase.sin() >= threshold

    site_pos = term._robot.data.site_pos_w[:, term._site_ids, :]
    r_lf = site_pos[:, term._lf_site_local_idx, :2]
    r_rf = site_pos[:, term._rf_site_local_idx, :2]

    lf_err = (r_lf - term._lf_landing_target[:, :2]).pow(2).sum(dim=-1)
    rf_err = (r_rf - term._rf_landing_target[:, :2]).pow(2).sum(dim=-1)

    lf_reward = torch.exp(-lf_err / (sigma ** 2))
    rf_reward = torch.exp(-rf_err / (sigma ** 2))

    lf_active = lf_in_swing & term._lf_landing_valid
    rf_active = rf_in_swing & term._rf_landing_valid
    n_active = lf_active.float() + rf_active.float()

    total = lf_active.float() * lf_reward + rf_active.float() * rf_reward
    reward = torch.where(n_active > 0, total / n_active,
                         torch.ones(B, device=device))

    vel_cmd = env.command_manager.get_command(term.cfg.vel_cmd_name)
    if vel_cmd is not None:
        cmd_mag = vel_cmd[:, :2].norm(dim=1) + vel_cmd[:, 2].abs()
        reward = torch.where(cmd_mag > command_threshold, reward,
                             torch.ones(B, device=device))

    return reward

def cop_forward_reward(
    env: "ManagerBasedRlEnv",
    heel_sensor_name: str = "heel_ground_contact",
    toe_sensor_name: str = "toe_ground_contact",
    target_ratio: float = 0.5,
    sigma: float = 0.25,
) -> Tensor:
    """Reward forward Center-of-Pressure via heel vs toe force distribution."""
    heel_sensor: ContactSensor = env.scene[heel_sensor_name]
    toe_sensor: ContactSensor = env.scene[toe_sensor_name]

    if heel_sensor.data.force is None or toe_sensor.data.force is None:
        return torch.ones(env.num_envs, device=env.device)

    heel_fz = heel_sensor.data.force[:, :, 2]
    toe_fz = toe_sensor.data.force[:, :, 2]

    left_heel_fz = heel_fz[:, :3].sum(dim=-1)
    right_heel_fz = heel_fz[:, 3:].sum(dim=-1)
    left_toe_fz = toe_fz[:, :4].sum(dim=-1)
    right_toe_fz = toe_fz[:, 4:].sum(dim=-1)

    eps = 1.0
    left_total = left_heel_fz + left_toe_fz
    right_total = right_heel_fz + right_toe_fz

    left_in_contact = left_total > eps
    right_in_contact = right_total > eps

    left_ratio = left_toe_fz / left_total.clamp(min=eps)
    right_ratio = right_toe_fz / right_total.clamp(min=eps)

    left_err = left_ratio - target_ratio
    right_err = right_ratio - target_ratio
    left_asym = torch.where(left_err < 0, 2.0 * left_err, left_err)
    right_asym = torch.where(right_err < 0, 2.0 * right_err, right_err)

    left_reward = torch.exp(-left_asym.pow(2) / (sigma ** 2))
    right_reward = torch.exp(-right_asym.pow(2) / (sigma ** 2))

    left_reward = torch.where(left_in_contact, left_reward,
                              torch.ones_like(left_reward))
    right_reward = torch.where(right_in_contact, right_reward,
                               torch.ones_like(right_reward))

    return 0.5 * (left_reward + right_reward)

@dataclass(kw_only=True)
class LocoManipMPCCommandCfg(LocoMPCCommandCfg):
    """Configuration for :class:`LocoManipMPCCommand`."""

    hand_site_names: tuple[str, str] = ("left_hand", "right_hand")

    lhand_box_sensor_name: str = "lhand_box_contact"
    rhand_box_sensor_name: str = "rhand_box_contact"

    body_box_sensor_name: str = ""

    box_mass: float = 8.0
    mu_ground: float = 0.5

    box_name: str = "box"
    K_box_vel: float = 5.0

    mu_hand: float = 0.6
    f_hand_max: float = 300.0
    R_f_hand: float = 1e-4
    R_hand_balance: float = 1e-3

    def build(self, env: "ManagerBasedRlEnv") -> "LocoManipMPCCommand":
        return LocoManipMPCCommand(self, env)

class LocoManipMPCCommand(LocoMPCCommand):
    """Loco-manipulation MPC runner — extends :class:`LocoMPCCommand`."""

    cfg: LocoManipMPCCommandCfg

    def __init__(self, cfg: LocoManipMPCCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        manip_mpc_cfg = LocoManipMPCConfig(
            N=cfg.mpc_horizon,
            dt=cfg.mpc_dt,
            mass=cfg.mass,
            mu_hand=cfg.mu_hand,
            f_hand_max=cfg.f_hand_max,
            R_f_hand=cfg.R_f_hand,
            R_hand_balance=cfg.R_hand_balance,
            solver_type=cfg.solver_type,
        )
        self._mpc = LocoManipMPC(manip_mpc_cfg, device=env.device)

        B = env.num_envs

        self._u_prev = torch.zeros(B, 18, device=env.device)

        self._hand_ref: Tensor = torch.zeros(B, 6, device=env.device)

        self._vis_r_lh: Tensor = torch.zeros(B, 3, device=env.device)
        self._vis_r_rh: Tensor = torch.zeros(B, 3, device=env.device)
        self._vis_f_lh: Tensor = torch.zeros(B, 3, device=env.device)
        self._vis_f_rh: Tensor = torch.zeros(B, 3, device=env.device)

        hand_site_cfg = SceneEntityCfg("robot", site_names=cfg.hand_site_names)
        hand_site_cfg.resolve(env.scene)
        self._hand_site_ids: list[int] = hand_site_cfg.site_ids

        _, _hand_names = self._robot.find_sites(
            cfg.hand_site_names, preserve_order=False
        )
        self._lh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "left" in n.lower()
        )
        self._rh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "right" in n.lower()
        )

        self._lhand_sensor: ContactSensor = env.scene[cfg.lhand_box_sensor_name]
        self._rhand_sensor: ContactSensor = env.scene[cfg.rhand_box_sensor_name]
        self._body_sensor: ContactSensor | None = (
            env.scene[cfg.body_box_sensor_name]
            if cfg.body_box_sensor_name
            else None
        )

        self._box_entity = env.scene[cfg.box_name]

        local_body_ids, _ = self._box_entity.find_bodies(
            cfg.box_name, preserve_order=True
        )
        self._box_body_global_id: int = int(
            self._box_entity.indexing.body_ids[local_body_ids[0]]
        )

        N_cfg = cfg.mpc_horizon
        self._vis_box_pos:      Tensor = torch.zeros(B, 3, device=env.device)
        self._vis_box_ref_traj: Tensor = torch.zeros(B, N_cfg + 1, 3, device=env.device)

    @property
    def command(self) -> Tensor:
        """Optimal first-step foot forces [B, 6]: [fLF(3), fRF(3)]."""
        return self._grf_ref

    def _update_command(self) -> None:
        """Solve the loco-manipulation MPC and update GRF + hand force refs."""
        self._step_count += 1
        self._traj_step  += 1
        self._interpolate_traj_refs()

        if self._step_count % self.cfg.run_every_n_steps != 0:
            return

        cfg    = self.cfg
        B      = self.num_envs
        N      = cfg.mpc_horizon
        dt     = cfg.mpc_dt
        device = self.device
        robot  = self._robot

        c  = robot.data.root_link_pos_w
        lv = robot.data.root_link_lin_vel_w
        av = robot.data.root_link_ang_vel_w
        l  = lv * cfg.mass
        k  = av @ self._I_approx
        x0 = torch.cat([c, l, k], dim=-1)

        site_pos  = robot.data.site_pos_w[:, self._site_ids, :]
        site_quat = robot.data.site_quat_w[:, self._site_ids, :]
        r_lf = site_pos[:, self._lf_site_local_idx, :]
        r_rf = site_pos[:, self._rf_site_local_idx, :]
        R_lf = _quat_to_rot(site_quat[:, self._lf_site_local_idx, :])
        R_rf = _quat_to_rot(site_quat[:, self._rf_site_local_idx, :])

        hand_pos  = robot.data.site_pos_w[:, self._hand_site_ids, :]
        hand_quat = robot.data.site_quat_w[:, self._hand_site_ids, :]
        r_lh = hand_pos[:, self._lh_local_idx, :]
        r_rh = hand_pos[:, self._rh_local_idx, :]
        R_lh = _quat_to_rot(hand_quat[:, self._lh_local_idx, :])
        R_rh = _quat_to_rot(hand_quat[:, self._rh_local_idx, :])

        vel_cmd = self._env.command_manager.get_command(cfg.vel_cmd_name)
        if vel_cmd is not None:
            vx_body = vel_cmd[:, 0]
            vy_body = vel_cmd[:, 1]
            wz      = vel_cmd[:, 2]
        else:
            vx_body = torch.zeros(B, device=device)
            vy_body = torch.zeros(B, device=device)
            wz      = torch.zeros(B, device=device)

        quat_w = robot.data.root_link_quat_w
        q_w = quat_w[:, 0]; q_x = quat_w[:, 1]
        q_y = quat_w[:, 2]; q_z = quat_w[:, 3]
        yaw   = torch.atan2(2.0 * (q_w * q_z + q_x * q_y),
                             1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        cos_y = yaw.cos(); sin_y = yaw.sin()
        vx    = cos_y * vx_body - sin_y * vy_body
        vy    = sin_y * vx_body + cos_y * vy_body

        k_steps = torch.arange(N + 1, device=device, dtype=torch.float32)
        yaw_k   = yaw.unsqueeze(1) + wz.unsqueeze(1) * k_steps * dt
        cos_k   = yaw_k.cos(); sin_k = yaw_k.sin()
        vx_w_k  = cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1)
        vy_w_k  = sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1)

        x_ref = x0.unsqueeze(1).expand(B, N + 1, -1).clone()
        x_ref[:, 0, 0] = c[:, 0]; x_ref[:, 0, 1] = c[:, 1]
        x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w_k[:, :-1] * dt, dim=1)
        x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w_k[:, :-1] * dt, dim=1)
        x_ref[:, :, 2]  = robot.data.default_root_state[:, 2:3]
        x_ref[:, :, 3]  = vx_w_k * cfg.mass
        x_ref[:, :, 4]  = vy_w_k * cfg.mass
        x_ref[:, :, 5]  = 0.0
        I_zz = float(self._I_approx[2, 2])
        x_ref[:, :, 6:8] = 0.0
        x_ref[:, :, 8]  = (I_zz * wz).unsqueeze(1)

        gait_phase = getattr(self._env, "_gait_phase",
                             torch.zeros(B, device=device))
        v_cmd_3d = torch.zeros(B, 3, device=device)
        v_cmd_3d[:, 0] = vx
        v_cmd_3d[:, 1] = vy
        schedule = make_walking_schedule(
            B=B, N=N, r_LF=r_lf, r_RF=r_rf,
            gait_phase=gait_phase, period=cfg.gait_period,
            dt=dt, duty_factor=cfg.duty_factor,
            com_pos=c, v_cmd=v_cmd_3d, yaw=yaw, yaw_rate=wz,
            hip_width=cfg.hip_width,
            R_LF_rot=R_lf, R_RF_rot=R_rf, device=device,
        )

        lh_found = self._lhand_sensor.data.found
        rh_found = self._rhand_sensor.data.found
        lh_active = ((lh_found > 0).any(dim=-1).float()
                     if lh_found is not None
                     else torch.zeros(B, device=device))
        rh_active = ((rh_found > 0).any(dim=-1).float()
                     if rh_found is not None
                     else torch.zeros(B, device=device))
        hand_contact = torch.stack([lh_active, rh_active], dim=-1)

        other_body_force  = torch.zeros(B, 3, device=device)
        other_body_torque = torch.zeros(B, 3, device=device)

        if self._body_sensor is not None and self._body_sensor.data.force is not None:
            body_net = self._body_sensor.data.force.sum(dim=1)

            lh_force = (self._lhand_sensor.data.force.sum(dim=1) * lh_active.unsqueeze(-1)
                        if self._lhand_sensor.data.force is not None
                        else torch.zeros(B, 3, device=device))
            rh_force = (self._rhand_sensor.data.force.sum(dim=1) * rh_active.unsqueeze(-1)
                        if self._rhand_sensor.data.force is not None
                        else torch.zeros(B, 3, device=device))

            other_body_force = body_net - lh_force - rh_force

        box_mass_per_env = self._env.sim.model.body_mass[:, self._box_body_global_id]
        push_xy_speed = (vx.pow(2) + vy.pow(2)).sqrt().clamp(min=1e-6)
        is_pushing    = (push_xy_speed > 0.05).float()
        resist_mag    = cfg.mu_ground * box_mass_per_env * 9.81
        box_resist_force = torch.stack([
            -is_pushing * resist_mag * (vx / push_xy_speed),
            -is_pushing * resist_mag * (vy / push_xy_speed),
            torch.zeros(B, device=device),
        ], dim=-1)

        u_ref  = torch.zeros(B, N, 18, device=device)
        mpc_in = LocoManipMPCInput(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            c_bar=None,
            R_LH=R_lh,
            R_RH=R_rh,
            r_LH=r_lh,
            r_RH=r_rh,
            hand_contact=hand_contact,
            other_body_force=other_body_force,
            other_body_torque=other_body_torque,
            box_resist_force=box_resist_force,
        )
        with torch.no_grad():
            mpc_out = self._mpc.solve(mpc_in)

        self._grf_ref = torch.cat([
            mpc_out.u_star[:, 0:3],
            mpc_out.u_star[:, 6:9],
        ], dim=-1)
        box_pos_w = self._box_entity.data.root_link_pos_w
        box_vel_w = self._box_entity.data.root_link_lin_vel_w

        box_ref = torch.zeros(B, N + 1, 3, device=device)
        box_ref[:, 0, 0] = box_pos_w[:, 0]
        box_ref[:, 0, 1] = box_pos_w[:, 1]
        box_ref[:, 0, 2] = box_pos_w[:, 2]
        box_ref[:, 1:, 0] = box_pos_w[:, 0:1] + torch.cumsum(vx_w_k[:, :N] * dt, dim=1)
        box_ref[:, 1:, 1] = box_pos_w[:, 1:2] + torch.cumsum(vy_w_k[:, :N] * dt, dim=1)
        box_ref[:, 1:, 2] = box_pos_w[:, 2:3]

        push_dir_x = is_pushing * (vx / push_xy_speed)
        push_dir_y = is_pushing * (vy / push_xy_speed)
        v_err_x = (vx - box_vel_w[:, 0]) * is_pushing
        v_err_y = (vy - box_vel_w[:, 1]) * is_pushing
        f_total_x = box_mass_per_env * cfg.K_box_vel * v_err_x + resist_mag * push_dir_x
        f_total_y = box_mass_per_env * cfg.K_box_vel * v_err_y + resist_mag * push_dir_y

        n_active     = (lh_active + rh_active).clamp(min=1.0)
        f_per_hand_x = f_total_x / n_active
        f_per_hand_y = f_total_y / n_active
        zeros_B = torch.zeros(B, device=device)
        lh_f_ref = torch.stack([
            f_per_hand_x * lh_active,
            f_per_hand_y * lh_active,
            zeros_B,
        ], dim=-1)
        rh_f_ref = torch.stack([
            f_per_hand_x * rh_active,
            f_per_hand_y * rh_active,
            zeros_B,
        ], dim=-1)
        self._hand_ref = torch.cat([lh_f_ref, rh_f_ref], dim=-1)

        self._u_prev = mpc_out.u_star.detach()

        x_pred = mpc_out.x_pred.detach()
        self._com_traj = x_pred[:, :, 0:3]
        l_traj = x_pred[:, :, 3:6]
        self._vel_traj = l_traj / cfg.mass
        k_pred = x_pred[:, :, 6:9]
        self._k_traj = k_pred
        self._kdot_traj = torch.zeros_like(k_pred)
        if N >= 2:
            self._kdot_traj[:, :-1] = (k_pred[:, 1:] - k_pred[:, :-1]) / dt
            self._kdot_traj[:, -1]  = self._kdot_traj[:, -2]

        self._traj_step = 0
        self._interpolate_traj_refs()

        self._vis_com      = self._com_traj
        self._vis_ang_mom  = self._k_traj
        self._vis_r_lf     = schedule.r_LF.detach()
        self._vis_r_rf     = schedule.r_RF.detach()
        self._vis_sigma_lf = schedule.sigma[:, :, 0].detach()
        self._vis_sigma_rf = schedule.sigma[:, :, 1].detach()
        self._vis_r_lh = r_lh.detach().clone()
        self._vis_r_rh = r_rh.detach().clone()
        self._vis_f_lh = lh_f_ref.detach().clone()
        self._vis_f_rh = rh_f_ref.detach().clone()
        self._vis_box_pos      = box_pos_w.detach().clone()
        self._vis_box_ref_traj = box_ref.detach().clone()

        sigma_lf = schedule.sigma[:, :, 0]
        sigma_rf = schedule.sigma[:, :, 1]
        lf_prev  = torch.cat([sigma_lf[:, :1], sigma_lf[:, :-1]], dim=1)
        rf_prev  = torch.cat([sigma_rf[:, :1], sigma_rf[:, :-1]], dim=1)
        lf_td    = (sigma_lf > 0.5) & (lf_prev < 0.5)
        rf_td    = (sigma_rf > 0.5) & (rf_prev < 0.5)
        self._lf_landing_valid = lf_td.any(dim=1)
        self._rf_landing_valid = rf_td.any(dim=1)
        lf_idx   = lf_td.float().argmax(dim=1)
        rf_idx   = rf_td.float().argmax(dim=1)
        batch_idx = torch.arange(B, device=device)
        self._lf_landing_target = torch.where(
            self._lf_landing_valid.unsqueeze(-1),
            schedule.r_LF[batch_idx, lf_idx],
            r_lf,
        )
        self._rf_landing_target = torch.where(
            self._rf_landing_valid.unsqueeze(-1),
            schedule.r_RF[batch_idx, rf_idx],
            r_rf,
        )

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Extend parent vis with hand force arrows and CoM reference spheres."""
        import numpy as np

        super()._debug_vis_impl(visualizer)

        B = self.num_envs
        env_indices = visualizer.get_env_indices(B)
        if not env_indices:
            return

        r_lh_np = self._vis_r_lh.cpu().numpy()
        r_rh_np = self._vis_r_rh.cpu().numpy()
        f_lh_np = self._vis_f_lh.cpu().numpy()
        f_rh_np = self._vis_f_rh.cpu().numpy()
        box_pos_np      = self._vis_box_pos.cpu().numpy()
        box_ref_traj_np = self._vis_box_ref_traj.cpu().numpy()
        N_pts = box_ref_traj_np.shape[1]

        _FORCE_SCALE = 0.002
        _F_MIN_N     = 5.0

        for b in env_indices:
            for side, r_np, f_np, rgb in (
                ("lh", r_lh_np[b], f_lh_np[b], (0.15, 0.80, 0.30)),
                ("rh", r_rh_np[b], f_rh_np[b], (1.00, 0.55, 0.10)),
            ):
                f_mag = float(np.linalg.norm(f_np))

                visualizer.add_sphere(
                    center=r_np,
                    radius=0.04,
                    color=(*rgb, 0.85),
                    label=f"mpc_hand_{side}_{b}",
                )

                if f_mag > _F_MIN_N:
                    tip = r_np + _FORCE_SCALE * f_np
                    visualizer.add_arrow(
                        start=r_np,
                        end=tip,
                        color=(*rgb, 0.95),
                        width=0.018,
                        label=f"mpc_hand_{side}_force_{b}",
                    )

            box_center = box_pos_np[b] + np.array([0.0, 0.0, 0.05])
            visualizer.add_sphere(
                center=box_center,
                radius=0.08,
                color=(0.9, 0.1, 0.1, 0.85),
                label=f"box_actual_{b}",
            )
            step_k = max(1, N_pts // 6)
            for k in range(0, N_pts, step_k):
                alpha = 0.30 + 0.65 * k / max(N_pts - 1, 1)
                visualizer.add_sphere(
                    center=box_ref_traj_np[b, k] + np.array([0.0, 0.0, 0.05]),
                    radius=0.035,
                    color=(1.0, 0.85, 0.0, alpha),
                    label=f"box_ref_{b}_{k}",
                )

    def _resample_command(self, env_ids: Tensor) -> None:
        """Reset all buffers for episode-terminated environments."""
        super()._resample_command(env_ids)
        self._hand_ref[env_ids] = 0.0
        self._u_prev[env_ids] = 0.0
        self._vis_box_pos[env_ids]      = 0.0
        self._vis_box_ref_traj[env_ids] = 0.0

def mpc_hand_force_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    lhand_sensor_name: str = "lhand_box_contact",
    rhand_sensor_name: str = "rhand_box_contact",
    sigma: float = 50.0,
) -> Tensor:
    """Reward the robot for tracking the MPC-optimal hand push forces."""
    from themis_training.push_box_mdp import _push_mask  # noqa: PLC0415

    term = env.command_manager.get_term(command_name)
    if term is None or not hasattr(term, "_hand_ref"):
        return torch.zeros(env.num_envs, device=env.device)

    hand_ref: Tensor = term._hand_ref
    B = env.num_envs

    lh_sensor: ContactSensor = env.scene[lhand_sensor_name]
    rh_sensor: ContactSensor = env.scene[rhand_sensor_name]

    def _get_force(sensor: ContactSensor) -> Tensor:
        if sensor.data.force is None:
            return torch.zeros(B, 3, device=env.device)
        return sensor.data.force.sum(dim=1)

    f_lh_actual = _get_force(lh_sensor)
    f_rh_actual = _get_force(rh_sensor)

    f_lh_ref = hand_ref[:, :3]
    f_rh_ref = hand_ref[:, 3:]

    ref_min_N = 5.0

    def _hand_reward(f_ref: Tensor, f_actual: Tensor) -> Tensor:
        want_push = f_ref.norm(dim=-1) > ref_min_N
        err_sq    = (f_actual - f_ref).pow(2).sum(dim=-1)
        r_track   = torch.exp(-err_sq / (sigma ** 2))
        return torch.where(want_push, r_track, f_ref.new_ones(B))

    r_lh = _hand_reward(f_lh_ref, f_lh_actual)
    r_rh = _hand_reward(f_rh_ref, f_rh_actual)

    push_mask = _push_mask(env).float()
    return 0.5 * (r_lh + r_rh) * push_mask
