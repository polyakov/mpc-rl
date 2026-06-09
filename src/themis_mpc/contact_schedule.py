"""Contact schedule utilities for multi-contact centroidal MPC."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

class ContactID(IntEnum):
    """Index ordering for the four contact end-effectors."""
    LF = 0
    RF = 1
    LH = 2
    RH = 3

@dataclass
class ContactSchedule:
    """Fixed contact schedule over the MPC horizon."""

    sigma: Tensor
    r_LF: Tensor
    r_RF: Tensor
    r_LH: Tensor
    r_RH: Tensor

    R_LF: Tensor | None = None
    R_RF: Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.sigma.device

    @property
    def batch_size(self) -> int:
        return self.sigma.shape[0]

    @property
    def horizon(self) -> int:
        return self.sigma.shape[1]

def make_double_support_schedule(
    B: int,
    N: int,
    r_LF: Tensor,
    r_RF: Tensor,
    r_LH: Tensor | None = None,
    r_RH: Tensor | None = None,
    R_LF_rot: Tensor | None = None,
    R_RF_rot: Tensor | None = None,
    hands_active: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ContactSchedule:
    """Create a simple double-support (both feet) schedule."""
    sigma = torch.zeros(B, N, 4, device=device, dtype=dtype)
    sigma[:, :, ContactID.LF] = 1.0
    sigma[:, :, ContactID.RF] = 1.0
    if hands_active:
        sigma[:, :, ContactID.LH] = 1.0
        sigma[:, :, ContactID.RH] = 1.0

    def _expand(t: Tensor | None, default_val: float = 0.0) -> Tensor:
        if t is None:
            return torch.full((B, N, 3), default_val, device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(B, -1)
        return t.unsqueeze(1).expand(B, N, 3)

    def _expand_rot(R: Tensor | None) -> Tensor | None:
        if R is None:
            return None
        R = R.to(device=device, dtype=dtype)
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(B, -1, -1)
        return R.contiguous()

    return ContactSchedule(
        sigma=sigma,
        r_LF=_expand(r_LF),
        r_RF=_expand(r_RF),
        r_LH=_expand(r_LH),
        r_RH=_expand(r_RH),
        R_LF=_expand_rot(R_LF_rot),
        R_RF=_expand_rot(R_RF_rot),
    )

def make_walking_schedule(
    B: int,
    N: int,
    r_LF: Tensor,
    r_RF: Tensor,
    gait_phase: Tensor,
    period: float = 0.7,
    dt: float = 0.05,
    duty_factor: float = 0.5,
    com_pos: "Tensor | None" = None,
    v_cmd: "Tensor | None" = None,
    yaw: "Tensor | None" = None,
    yaw_rate: "Tensor | None" = None,
    hip_width: float = 0.1,
    R_LF_rot: "Tensor | None" = None,
    R_RF_rot: "Tensor | None" = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ContactSchedule:
    """Create a walking schedule with predictive Raibert-style foot placement."""
    import math

    r_LF = r_LF.to(device=device, dtype=dtype)
    r_RF = r_RF.to(device=device, dtype=dtype)

    phase_rate = 2.0 * math.pi / period
    k_steps = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)
    phase_traj = gait_phase.unsqueeze(1) + phase_rate * dt * k_steps

    threshold = math.cos(math.pi * duty_factor)
    rf_stance = phase_traj.sin() < threshold
    lf_stance = (phase_traj + math.pi).sin() < threshold

    sigma = torch.zeros(B, N, 4, device=device, dtype=dtype)
    sigma[:, :, ContactID.LF] = lf_stance.to(dtype)
    sigma[:, :, ContactID.RF] = rf_stance.to(dtype)

    if com_pos is not None and v_cmd is not None:
        v_cmd = v_cmd.to(device=device, dtype=dtype)
        com_pos = com_pos.to(device=device, dtype=dtype)

        stride_time = period / 2.0

        wz_t = (yaw_rate.to(device=device, dtype=dtype)
                if yaw_rate is not None
                else torch.zeros(B, device=device, dtype=dtype))
        yaw_t = (yaw.to(device=device, dtype=dtype)
                 if yaw is not None
                 else torch.zeros(B, device=device, dtype=dtype))

        cos_y0 = yaw_t.cos(); sin_y0 = yaw_t.sin()
        vx_body = ( cos_y0 * v_cmd[:, 0] + sin_y0 * v_cmd[:, 1])
        vy_body = (-sin_y0 * v_cmd[:, 0] + cos_y0 * v_cmd[:, 1])

        k_idx   = torch.arange(N, device=device, dtype=dtype)
        yaw_k   = yaw_t.unsqueeze(1) + wz_t.unsqueeze(1) * k_idx * dt
        cos_k   = yaw_k.cos()
        sin_k   = yaw_k.sin()

        vx_k = (cos_k * vx_body.unsqueeze(1) - sin_k * vy_body.unsqueeze(1))
        vy_k = (sin_k * vx_body.unsqueeze(1) + cos_k * vy_body.unsqueeze(1))

        com_arc = torch.zeros(B, N, 3, device=device, dtype=dtype)
        com_arc[:, :, 0] = com_pos[:, 0:1] + torch.cumsum(vx_k * dt, dim=1)
        com_arc[:, :, 1] = com_pos[:, 1:2] + torch.cumsum(vy_k * dt, dim=1)
        com_arc[:, :, 2] = com_pos[:, 2:3].expand(B, N)

        def _foot_traj(r_f0: Tensor, stance_mask: Tensor, sign_y: float) -> Tensor:
            """Propagate foot position; apply Raibert heuristic at touchdowns."""
            traj = torch.zeros(B, N, 3, device=device, dtype=dtype)
            r_cur = r_f0.clone()
            r_f0_z = r_f0[:, 2].clamp(min=0.0)

            for k in range(N):
                is_stance = stance_mask[:, k]
                new_stance = (
                    is_stance & (~stance_mask[:, k - 1])
                    if k > 0
                    else torch.zeros(B, device=device, dtype=torch.bool)
                )

                if new_stance.any():
                    hip_x = sign_y * (-sin_k[:, k]) * hip_width
                    hip_y = sign_y * ( cos_k[:, k]) * hip_width

                    p_new_x = (com_arc[:, k, 0]
                                + vx_k[:, k] * 0.5 * stride_time
                                + hip_x)
                    p_new_y = (com_arc[:, k, 1]
                                + vy_k[:, k] * 0.5 * stride_time
                                + hip_y)
                    p_new = torch.stack([p_new_x, p_new_y, r_f0_z], dim=-1)
                    r_cur = torch.where(new_stance.unsqueeze(-1), p_new, r_cur)

                traj[:, k, :] = r_cur

            return traj

        r_LF_traj = _foot_traj(r_LF, lf_stance, sign_y=+1.0)
        r_RF_traj = _foot_traj(r_RF, rf_stance, sign_y=-1.0)
    else:
        r_LF_traj = r_LF.unsqueeze(1).expand(B, N, 3).contiguous()
        r_RF_traj = r_RF.unsqueeze(1).expand(B, N, 3).contiguous()

    def _expand_rot(R: "Tensor | None") -> "Tensor | None":
        if R is None:
            return None
        R = R.to(device=device, dtype=dtype)
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(B, -1, -1)
        return R.contiguous()

    return ContactSchedule(
        sigma=sigma,
        r_LF=r_LF_traj,
        r_RF=r_RF_traj,
        r_LH=torch.zeros(B, N, 3, device=device, dtype=dtype),
        r_RH=torch.zeros(B, N, 3, device=device, dtype=dtype),
        R_LF=_expand_rot(R_LF_rot),
        R_RF=_expand_rot(R_RF_rot),
    )
