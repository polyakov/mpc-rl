"""Loco-manipulation centroidal MPC (locomotion + hand push forces)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from themis_mpc.admm_qp import QPData, QPSolution
from themis_mpc.centroidal_mpc import (
    CentroidalMPC,
    MPCConfig,
    MPCOutput,
    _jax_to_torch,
    _load_jax_pimpc,
    _skew,
    _torch_to_jax,
)
from themis_mpc.contact_schedule import ContactSchedule

@dataclass
class LocoManipMPCConfig(MPCConfig):
    """Loco-manipulation MPC configuration."""

    mu_hand: float = 0.6
    f_hand_max: float = 300.0

    R_f_hand: float = 1e-4
    R_hand_balance: float = 1e-3

    use_condensed: bool = False

@dataclass
class LocoManipMPCInput:
    """MPC input for the loco-manipulation solver."""
    x0:       Tensor
    schedule: ContactSchedule
    x_ref:    Tensor
    u_ref:    Tensor
    u_prev:   Tensor
    c_bar:    Tensor | None

    R_LH: Tensor
    R_RH: Tensor
    r_LH: Tensor
    r_RH: Tensor

    hand_contact: Tensor

    other_body_force:  Tensor
    other_body_torque: Tensor
    box_resist_force:  Tensor

class LocoManipMPC(CentroidalMPC):
    """Contact-scheduled centroidal QP-MPC with hand push forces."""

    _nu: int = 18
    _nx: int = 9

    def __init__(self, cfg: LocoManipMPCConfig, device: torch.device | str = "cpu"):
        super().__init__(cfg, device)

    def _build_constants(self) -> None:
        """Rebuild all constant matrices for nu = 18."""
        cfg: LocoManipMPCConfig = self.cfg  # type: ignore[assignment]
        dt, m = cfg.dt, cfg.mass
        g  = torch.tensor(cfg.gravity, device=self.device, dtype=self.dtype)
        I3 = torch.eye(3, device=self.device, dtype=self.dtype)
        nx, nu, N = self._nx, self._nu, cfg.N

        self.A = torch.zeros(nx, nx, device=self.device, dtype=self.dtype)
        self.A[:3, :3]   = I3
        self.A[:3, 3:6]  = (dt / m) * I3
        self.A[3:6, 3:6] = I3
        self.A[6:9, 6:9] = I3

        self.d = torch.zeros(nx, device=self.device, dtype=self.dtype)
        self.d[:3]  = 0.5 * dt**2 * g
        self.d[3:6] = m * dt * g

        Q_diag = torch.tensor(
            list(cfg.Q_c) + list(cfg.Q_l) + list(cfg.Q_k),
            device=self.device, dtype=self.dtype,
        )
        self.Q  = torch.diag(Q_diag)
        self.Qf = cfg.Qf_scale * self.Q

        R_diag = torch.zeros(nu, device=self.device, dtype=self.dtype)
        R_diag[0:3]  = cfg.R_f_foot
        R_diag[3:6]  = cfg.R_tau_foot
        R_diag[6:9]  = cfg.R_f_foot
        R_diag[9:12] = cfg.R_tau_foot
        R_diag[12:15] = cfg.R_f_hand
        R_diag[15:18] = cfg.R_f_hand

        D_bal = torch.zeros(3, nu, device=self.device, dtype=self.dtype)
        D_bal[:, 12:15] =  I3
        D_bal[:, 15:18] = -I3
        H_balance = cfg.R_hand_balance * D_bal.t().mm(D_bal)

        self.R       = torch.diag(R_diag) + H_balance
        self.R_delta = cfg.R_delta * torch.eye(nu, device=self.device, dtype=self.dtype)

        Q_bar = torch.zeros(nx*N, nx*N, device=self.device, dtype=self.dtype)
        for i in range(N - 1):
            s = i*nx; Q_bar[s:s+nx, s:s+nx] = self.Q
        s = (N-1)*nx; Q_bar[s:s+nx, s:s+nx] = self.Qf
        self._Q_bar = Q_bar

        R_bar       = torch.zeros(nu*N, nu*N, device=self.device, dtype=self.dtype)
        R_delta_bar = torch.zeros(nu*N, nu*N, device=self.device, dtype=self.dtype)
        D_diff      = torch.zeros(nu*N, nu*N, device=self.device, dtype=self.dtype)
        for i in range(N):
            s = i * nu
            R_bar[s:s+nu, s:s+nu]       = self.R
            R_delta_bar[s:s+nu, s:s+nu] = self.R_delta
            D_diff[s:s+nu, s:s+nu]      = torch.eye(nu, device=self.device, dtype=self.dtype)
            if i > 0:
                D_diff[s:s+nu, s-nu:s]  = -torch.eye(nu, device=self.device, dtype=self.dtype)

        self._R_bar       = R_bar
        self._R_delta_bar = R_delta_bar
        self._D_diff      = D_diff
        self._DtRd        = D_diff.t().mm(R_delta_bar)

        H_uu = R_bar + D_diff.t().mm(R_delta_bar.mm(D_diff))
        self._H_uu = H_uu

        n_z_sparse = (nx + nu) * N
        H_full = torch.zeros(n_z_sparse, n_z_sparse, device=self.device, dtype=self.dtype)
        H_full[:nx*N, :nx*N] = Q_bar
        H_full[nx*N:, nx*N:] = H_uu
        self._H_full = 0.5 * (H_full + H_full.t())

        mu    = cfg.mu_foot
        yh    = cfg.foot_y_half
        toe   = cfg.foot_x_toe
        heel  = cfg.foot_x_heel
        mu_z  = cfg.mu_foot_yaw
        fz_max = cfg.fz_max_foot
        foot_rows = [
            [ 1,  0,    -mu,   0,    0,  0],
            [-1,  0,    -mu,   0,    0,  0],
            [ 0,  1,    -mu,   0,    0,  0],
            [ 0, -1,    -mu,   0,    0,  0],
            [ 0,  0,     -1,   0,    0,  0],
            [ 0,  0,      1,   0,    0,  0],
            [ 0,  0,    -yh,   1,    0,  0],
            [ 0,  0,    -yh,  -1,    0,  0],
            [ 0,  0,  -heel,   0,    1,  0],
            [ 0,  0,   -toe,   0,   -1,  0],
            [ 0,  0,  -mu_z,   0,    0,  1],
            [ 0,  0,  -mu_z,   0,    0, -1],
        ]
        self._foot_cone_G = torch.tensor(foot_rows, device=self.device, dtype=self.dtype)
        self._foot_cone_b = torch.zeros(12, device=self.device, dtype=self.dtype)
        self._foot_cone_b[5] = fz_max

        mu_h = cfg.mu_hand
        f_h  = cfg.f_hand_max
        hand_rows = [
            [-mu_h,  1.0,  0.0],
            [-mu_h, -1.0,  0.0],
            [-mu_h,  0.0,  1.0],
            [-mu_h,  0.0, -1.0],
            [-1.0,   0.0,  0.0],
            [ 1.0,   0.0,  0.0],
        ]
        self._hand_cone_G = torch.tensor(hand_rows, device=self.device, dtype=self.dtype)
        self._hand_cone_b = torch.zeros(6, device=self.device, dtype=self.dtype)
        self._hand_cone_b[5] = f_h

    def _build_Bk_manip(self, mpc_in: LocoManipMPCInput) -> Tensor:
        """Build per-step input matrix B_k for nu = 18.  Returns (B, N, 9, 18)."""
        cfg = self.cfg
        schedule = mpc_in.schedule
        B_, N   = schedule.batch_size, schedule.horizon
        dt, m   = cfg.dt, cfg.mass
        device, dtype = self.device, self.dtype
        I3  = torch.eye(3, device=device, dtype=dtype)

        sigma = schedule.sigma
        r_LF  = schedule.r_LF
        r_RF  = schedule.r_RF

        s_LH_gate = mpc_in.hand_contact[:, 0].view(B_, 1, 1)
        s_RH_gate = mpc_in.hand_contact[:, 1].view(B_, 1, 1)

        r_LH_h = mpc_in.r_LH.unsqueeze(1).expand(B_, N, 3)
        r_RH_h = mpc_in.r_RH.unsqueeze(1).expand(B_, N, 3)

        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                 else mpc_in.x_ref[:, :N, :3])

        Bk = torch.zeros(B_, N, 9, 18, device=device, dtype=dtype)

        for k in range(N):
            c_k  = c_bar[:, k, :]
            s_LF = sigma[:, k, 0:1].unsqueeze(-1)
            s_RF = sigma[:, k, 1:2].unsqueeze(-1)

            Ef = torch.zeros(B_, 3, 18, device=device, dtype=dtype)
            Ef[:, :, 0:3]  = s_LF * I3.unsqueeze(0)
            Ef[:, :, 6:9]  = s_RF * I3.unsqueeze(0)
            Ef[:, :, 12:15] = s_LH_gate * I3.unsqueeze(0)
            Ef[:, :, 15:18] = s_RH_gate * I3.unsqueeze(0)

            Et = torch.zeros(B_, 3, 18, device=device, dtype=dtype)
            Et[:, :, 0:3]   = s_LF * _skew(r_LF[:, k, :] - c_k)
            Et[:, :, 3:6]   = s_LF * I3.unsqueeze(0)
            Et[:, :, 6:9]   = s_RF * _skew(r_RF[:, k, :] - c_k)
            Et[:, :, 9:12]  = s_RF * I3.unsqueeze(0)
            Et[:, :, 12:15] = s_LH_gate * _skew(r_LH_h[:, k, :] - c_k)
            Et[:, :, 15:18] = s_RH_gate * _skew(r_RH_h[:, k, :] - c_k)

            Bk[:, k, 0:3, :] = (dt**2 / (2*m)) * Ef
            Bk[:, k, 3:6, :] = dt * Ef
            Bk[:, k, 6:9, :] = dt * Et

        return Bk

    def _compute_c_extra(self, mpc_in: LocoManipMPCInput) -> Tensor:
        """Centroidal disturbance from non-hand contacts and box resistance."""
        cfg = self.cfg
        dt, m  = cfg.dt, cfg.mass
        device = self.device
        dtype  = self.dtype

        F_total = mpc_in.other_body_force + mpc_in.box_resist_force
        tau_other = mpc_in.other_body_torque

        B_ = mpc_in.x0.shape[0]
        c_extra = torch.zeros(B_, 9, device=device, dtype=dtype)
        c_extra[:, 0:3] = (dt**2 / (2 * m)) * F_total
        c_extra[:, 3:6] = dt * F_total
        c_extra[:, 6:9] = dt * tau_other
        return c_extra

    def _build_sparse_qp_manip(
        self, mpc_in: LocoManipMPCInput, Bk: Tensor
    ) -> QPData:
        """Sparse (x, u) QP for the loco-manipulation MPC."""
        cfg   = self.cfg
        N     = cfg.N
        B_    = mpc_in.x0.shape[0]
        nx    = self._nx
        nu    = self._nu
        n_z   = (nx + nu) * N
        xs, us = 0, nx * N
        device, dtype = self.device, self.dtype

        H = self._H_full.unsqueeze(0).expand(B_, -1, -1)

        X_ref  = mpc_in.x_ref[:, 1:, :].reshape(B_, nx*N)
        U_ref  = mpc_in.u_ref.reshape(B_, nu*N)
        e_prev = torch.zeros(B_, nu*N, device=device, dtype=dtype)
        e_prev[:, :nu] = mpc_in.u_prev

        h = torch.zeros(B_, n_z, device=device, dtype=dtype)
        h[:, xs:xs+nx*N] = -(self._Q_bar.unsqueeze(0) @ X_ref.unsqueeze(-1)).squeeze(-1)
        h[:, us:us+nu*N] = (
            -(self._R_bar.unsqueeze(0)  @ U_ref.unsqueeze(-1)).squeeze(-1)
            -(self._DtRd.unsqueeze(0)   @ e_prev.unsqueeze(-1)).squeeze(-1)
        )

        n_eq = nx * N
        A_eq = torch.zeros(B_, n_eq, n_z, device=device, dtype=dtype)
        b_eq = torch.zeros(B_, n_eq,      device=device, dtype=dtype)
        I_nx = torch.eye(nx, device=device, dtype=dtype)
        d_B  = self.d.unsqueeze(0).expand(B_, -1)
        rhs_0 = mpc_in.x0 @ self.A.t() + self.d

        dc = self._compute_c_extra(mpc_in)

        for k in range(N):
            rs, re = k*nx, (k+1)*nx
            A_eq[:, rs:re, xs+k*nx : xs+(k+1)*nx] = I_nx
            if k > 0:
                A_eq[:, rs:re, xs+(k-1)*nx : xs+k*nx] = -self.A
            A_eq[:, rs:re, us+k*nu : us+(k+1)*nu] = -Bk[:, k, :, :]
            b_eq[:, rs:re] = (rhs_0 + dc) if k == 0 else (d_B + dc)

        n_fc_foot = self._foot_cone_G.shape[0]
        n_fc_hand = self._hand_cone_G.shape[0]
        n_ineq_step = 2 * n_fc_foot + 2 * n_fc_hand
        n_ineq      = N * n_ineq_step

        G = torch.zeros(B_, n_ineq, n_z, device=device, dtype=dtype)
        b = torch.zeros(B_, n_ineq,      device=device, dtype=dtype)

        fc_G_base  = self._foot_cone_G
        fc_b_base  = self._foot_cone_b.unsqueeze(0).expand(B_, -1)
        hc_G_base  = self._hand_cone_G
        hc_b_base  = self._hand_cone_b.unsqueeze(0).expand(B_, -1)

        sch = mpc_in.schedule
        if sch.R_LF is not None and sch.R_RF is not None:
            from themis_mpc.centroidal_mpc import _make_T6
            T6_LF = _make_T6(sch.R_LF.to(device=device, dtype=dtype))
            T6_RF = _make_T6(sch.R_RF.to(device=device, dtype=dtype))
            fc_G_LF = torch.bmm(fc_G_base.unsqueeze(0).expand(B_, -1, -1), T6_LF)
            fc_G_RF = torch.bmm(fc_G_base.unsqueeze(0).expand(B_, -1, -1), T6_RF)
        else:
            fc_G_LF = fc_G_base.unsqueeze(0).expand(B_, -1, -1)
            fc_G_RF = fc_G_base.unsqueeze(0).expand(B_, -1, -1)

        T3_LH = mpc_in.R_LH.to(device=device, dtype=dtype).transpose(-1, -2)
        T3_RH = mpc_in.R_RH.to(device=device, dtype=dtype).transpose(-1, -2)
        hc_G_LH = torch.bmm(hc_G_base.unsqueeze(0).expand(B_, -1, -1), T3_LH)
        hc_G_RH = torch.bmm(hc_G_base.unsqueeze(0).expand(B_, -1, -1), T3_RH)

        for k in range(N):
            uo  = us + k * nu
            co  = k * n_ineq_step

            r0 = co
            r1 = co + n_fc_foot
            r2 = co + 2 * n_fc_foot
            r3 = co + 2 * n_fc_foot + n_fc_hand
            r4 = co + n_ineq_step

            G[:, r0:r1, uo:uo+6]    = fc_G_LF
            b[:, r0:r1]              = fc_b_base

            G[:, r1:r2, uo+6:uo+12] = fc_G_RF
            b[:, r1:r2]              = fc_b_base

            G[:, r2:r3, uo+12:uo+15] = hc_G_LH
            b[:, r2:r3]               = hc_b_base

            G[:, r3:r4, uo+15:uo+18] = hc_G_RH
            b[:, r3:r4]               = hc_b_base

        lb = torch.full((B_, n_z), -1e6, device=device, dtype=dtype)
        ub = torch.full((B_, n_z),  1e6, device=device, dtype=dtype)

        inactive_LH = mpc_in.hand_contact[:, 0] < 0.5
        inactive_RH = mpc_in.hand_contact[:, 1] < 0.5

        for k in range(N):
            uo      = us + k * nu
            sigma_k = mpc_in.schedule.sigma[:, k, :]

            inactive_LF = sigma_k[:, 0] < 0.5
            lb[inactive_LF, uo:uo+6]    = 0.0
            ub[inactive_LF, uo:uo+6]    = 0.0
            inactive_RF = sigma_k[:, 1] < 0.5
            lb[inactive_RF, uo+6:uo+12] = 0.0
            ub[inactive_RF, uo+6:uo+12] = 0.0

            lb[inactive_LH, uo+12:uo+15] = 0.0
            ub[inactive_LH, uo+12:uo+15] = 0.0
            lb[inactive_RH, uo+15:uo+18] = 0.0
            ub[inactive_RH, uo+15:uo+18] = 0.0

        return QPData(H=H, h=h, G=G, b=b, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq)

    def solve(self, mpc_in: LocoManipMPCInput) -> MPCOutput:  # type: ignore[override]
        """Solve the loco-manipulation MPC."""
        if self.cfg.use_condensed:
            raise NotImplementedError(
                "Condensed formulation is not supported for LocoManipMPC "
                "(nu=18 condensation is not implemented).  "
                "Set cfg.use_condensed=False."
            )
        if self.cfg.solver_type == "pimpc":
            return self._solve_pimpc_manip(mpc_in)
        if self.cfg.solver_type == "jax_pimpc":
            return self._solve_jax_pimpc_manip(mpc_in)
        return self._solve_sparse_manip(mpc_in)

    def _solve_pimpc_manip(self, mpc_in: LocoManipMPCInput) -> MPCOutput:
        """Solve via PiMPC with 18-D decision variable (feet + hands).

        Uses per-env, per-step linearised dynamics B_k (no batch/horizon
        averaging) and routes the per-env centroidal disturbance through the
        batched ``w`` term.  Preconditioning is applied externally (Ruiz), as
        the solver does not support internal preconditioning with per-env B.
        Mirrors :meth:`CentroidalMPC._solve_pimpc`.
        """
        cfg = self.cfg
        N       = cfg.N
        B_batch = mpc_in.x0.shape[0]
        nx, nu  = self._nx, self._nu
        do_profile = cfg.profile

        t0 = self._tick() if do_profile else 0.0

        Bk = self._build_Bk_manip(mpc_in)
        Bk = torch.nan_to_num(Bk, nan=0.0, posinf=0.0, neginf=0.0)

        dc = self._compute_c_extra(mpc_in)
        dc = torch.nan_to_num(dc, nan=0.0, posinf=0.0, neginf=0.0)

        mu     = cfg.mu_foot
        fz_max = cfg.fz_max_foot
        yh     = cfg.foot_y_half
        toe    = cfg.foot_x_toe
        heel   = cfg.foot_x_heel
        mu_z   = cfg.mu_foot_yaw
        mu_h   = cfg.mu_hand
        f_h    = cfg.f_hand_max

        umin = torch.tensor([
            -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            0.0, -mu_h*f_h, -mu_h*f_h,
            0.0, -mu_h*f_h, -mu_h*f_h,
        ], device=self.device, dtype=self.dtype)

        umax = torch.tensor([
            mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            f_h, mu_h*f_h, mu_h*f_h,
            f_h, mu_h*f_h, mu_h*f_h,
        ], device=self.device, dtype=self.dtype)

        yref = mpc_in.x_ref[:, 1:N+1, :].permute(0, 2, 1).contiguous()
        uref = mpc_in.u_ref.permute(0, 2, 1).contiguous()
        x0_in = mpc_in.x0
        u0_in = mpc_in.u_prev
        w_in  = dc

        sigma   = mpc_in.schedule.sigma
        lf_act  = sigma[:, :, 0:1].permute(0, 2, 1).float()
        rf_act  = sigma[:, :, 1:2].permute(0, 2, 1).float()
        lh_val  = mpc_in.hand_contact[:, 0:1].unsqueeze(-1)
        rh_val  = mpc_in.hand_contact[:, 1:2].unsqueeze(-1)
        lh_act  = lh_val.expand(B_batch, 1, N).float()
        rh_act  = rh_val.expand(B_batch, 1, N).float()

        umin_steps = umin.view(1, nu, 1).expand(B_batch, nu, N).clone()
        umax_steps = umax.view(1, nu, 1).expand(B_batch, nu, N).clone()
        umin_steps[:, :6,    :] *= lf_act
        umax_steps[:, :6,    :] *= lf_act
        umin_steps[:, 6:12,  :] *= rf_act
        umax_steps[:, 6:12,  :] *= rf_act
        umin_steps[:, 12:15, :] *= lh_act
        umax_steps[:, 12:15, :] *= lh_act
        umin_steps[:, 15:18, :] *= rh_act
        umax_steps[:, 15:18, :] *= rh_act

        if cfg.pimpc_precondition:
            s = self._get_pimpc_ruiz_cache(nu)
            A_in   = s["A_s"]
            e_in   = s["e_s"]
            Wy_in  = s["Wy_s"]
            Wf_in  = s["Wf_s"]
            Wu_in  = s["Wu_s"]
            Wdu_in = s["Wdu_s"]
            Bk_in   = torch.einsum("ij,bkjl,lm->bkim", s["Dxi"], Bk, s["Du"])
            x0_in   = x0_in @ s["Dxi"]
            u0_in   = u0_in @ s["Dui"]
            yref    = torch.einsum("ij,bjk->bik", s["Dxi"], yref)
            uref    = torch.einsum("ij,bjk->bik", s["Dui"], uref)
            umin_st = torch.einsum("ij,bjk->bik", s["Dui"], umin_steps)
            umax_st = torch.einsum("ij,bjk->bik", s["Dui"], umax_steps)
            umin_in = s["Dui"] @ umin
            umax_in = s["Dui"] @ umax
            w_in    = w_in @ s["Dxi"]
        else:
            A_in, e_in, Bk_in = self.A, self.d, Bk
            Wy_in, Wu_in, Wf_in = self.Q, self.R, self.Qf
            Wdu_in = cfg.R_delta * torch.eye(nu, device=self.device, dtype=self.dtype)
            umin_st, umax_st = umin_steps, umax_steps
            umin_in, umax_in = umin, umax

        self._pimpc.setup(
            A=A_in, B=Bk_in, Np=N,
            e=e_in,
            Wy=Wy_in, Wu=Wu_in,
            Wdu=Wdu_in,
            Wf=Wf_in,
            umin=umin_in, umax=umax_in,
            rho=cfg.pimpc_rho,
            maxiter=cfg.admm_max_iter,
            accel=cfg.pimpc_accel,
            tol=cfg.admm_eps_abs,
            device=str(self.device),
            dtype=self.dtype,
        )

        t1 = self._tick() if do_profile else 0.0

        try:
            result = self._pimpc.solve_batch(
                x0_in, u0_in, yref, uref, w=w_in,
                umin_steps=umin_st, umax_steps=umax_st,
            )
            x_pred = result.x[:, :, 1:].permute(0, 2, 1).contiguous()
            u_pred = result.u.permute(0, 2, 1).contiguous()
            if cfg.pimpc_precondition:
                x_pred = x_pred * s["dx"].view(1, 1, nx)
                u_pred = u_pred * s["du"].view(1, 1, nu)
            pimpc_solution = self._make_pimpc_solution(result, B_batch)
        except torch.linalg.LinAlgError as err:
            print(
                f"[LocoManipMPC] PiMPC solve failed ({err}); "
                f"falling back to reference + warm-start u_prev."
            )
            x_pred = mpc_in.x_ref[:, 1:N + 1, :].contiguous()
            u_pred = mpc_in.u_prev.unsqueeze(1).expand(B_batch, N, nu).contiguous()
            pimpc_solution = self._make_fallback_solution(B_batch)

        t2 = self._tick() if do_profile else 0.0

        x_pred = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)
        u_pred = torch.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)

        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms": t1 - t0, "solve_ms": t2 - t1, "recover_ms": t3 - t2,
            }

        return self._make_output_manip(
            mpc_in, x_pred, u_pred, pimpc_solution,
        )

    def _solve_jax_pimpc_manip(self, mpc_in: LocoManipMPCInput) -> MPCOutput:
        """Solve the 18-D loco-manip QP via the JAX PiMPC backend.

        Mirrors :meth:`_solve_pimpc_manip` (per-env per-step B_k, per-env
        centroidal disturbance, box-relaxed friction cones) but dispatches to
        the AOT-compiled JAX solver, which preconditions internally.
        """
        import jax.numpy as jnp

        cfg = self.cfg
        N       = cfg.N
        B_batch = mpc_in.x0.shape[0]
        nx, nu  = self._nx, self._nu
        do_profile = cfg.profile

        t0 = self._tick() if do_profile else 0.0

        Bk = self._build_Bk_manip(mpc_in)
        Bk = torch.nan_to_num(Bk, nan=0.0, posinf=0.0, neginf=0.0)
        dc = self._compute_c_extra(mpc_in)
        dc = torch.nan_to_num(dc, nan=0.0, posinf=0.0, neginf=0.0)
        e_env = self.d.unsqueeze(0) + dc  # [B, nx] per-env disturbance

        mu     = cfg.mu_foot
        fz_max = cfg.fz_max_foot
        yh     = cfg.foot_y_half
        toe    = cfg.foot_x_toe
        heel   = cfg.foot_x_heel
        mu_z   = cfg.mu_foot_yaw
        mu_h   = cfg.mu_hand
        f_h    = cfg.f_hand_max

        umin = torch.tensor([
            -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            0.0, -mu_h*f_h, -mu_h*f_h,
            0.0, -mu_h*f_h, -mu_h*f_h,
        ], device=self.device, dtype=self.dtype)
        umax = torch.tensor([
            mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            f_h, mu_h*f_h, mu_h*f_h,
            f_h, mu_h*f_h, mu_h*f_h,
        ], device=self.device, dtype=self.dtype)

        sigma   = mpc_in.schedule.sigma
        lf_act  = sigma[:, :, 0:1].permute(0, 2, 1).float()
        rf_act  = sigma[:, :, 1:2].permute(0, 2, 1).float()
        lh_val  = mpc_in.hand_contact[:, 0:1].unsqueeze(-1)
        rh_val  = mpc_in.hand_contact[:, 1:2].unsqueeze(-1)
        lh_act  = lh_val.expand(B_batch, 1, N).float()
        rh_act  = rh_val.expand(B_batch, 1, N).float()

        umin_steps = umin.view(1, nu, 1).expand(B_batch, nu, N).clone()
        umax_steps = umax.view(1, nu, 1).expand(B_batch, nu, N).clone()
        umin_steps[:, :6,    :] *= lf_act
        umax_steps[:, :6,    :] *= lf_act
        umin_steps[:, 6:12,  :] *= rf_act
        umax_steps[:, 6:12,  :] *= rf_act
        umin_steps[:, 12:15, :] *= lh_act
        umax_steps[:, 12:15, :] *= lh_act
        umin_steps[:, 15:18, :] *= rh_act
        umax_steps[:, 15:18, :] *= rh_act

        yref = mpc_in.x_ref[:, 1:N+1, :].permute(0, 2, 1).contiguous()
        uref = mpc_in.u_ref.permute(0, 2, 1).contiguous()
        Wdu  = cfg.R_delta * torch.eye(nu, device=self.device, dtype=self.dtype)

        jax_mod = _load_jax_pimpc()
        shape_key = (B_batch, N)
        if (self._jax_pimpc_solver is None
                or self._jax_pimpc_shape != shape_key):
            jax_dtype = jnp.float32 if self.dtype == torch.float32 else jnp.float64
            self._jax_pimpc_solver = jax_mod.PiMPCSolver(
                B=B_batch, N=N, nx=nx, nu=nu,
                maxiter=cfg.admm_max_iter,
                accel=cfg.pimpc_accel,
                precondition=cfg.pimpc_precondition,
                dtype=jax_dtype,
                compile_now=True,
            )
            self._jax_pimpc_shape = shape_key

        t1 = self._tick() if do_profile else 0.0

        try:
            prob = {
                "A":          _torch_to_jax(self.A),
                "B_s":        _torch_to_jax(Bk),
                "e":          _torch_to_jax(e_env),
                "Wy":         _torch_to_jax(self.Q),
                "Wu":         _torch_to_jax(self.R),
                "Wdu":        _torch_to_jax(Wdu),
                "Wf":         _torch_to_jax(self.Qf),
                "x0":         _torch_to_jax(mpc_in.x0),
                "u0":         _torch_to_jax(mpc_in.u_prev),
                "yref":       _torch_to_jax(yref),
                "uref":       _torch_to_jax(uref),
                "umin_steps": _torch_to_jax(umin_steps),
                "umax_steps": _torch_to_jax(umax_steps),
            }
            x_jax, u_jax, res_jax = self._jax_pimpc_solver.solve(
                prob, rho=cfg.pimpc_rho,
            )
            x_jax.block_until_ready()
            x_pred = _jax_to_torch(x_jax).to(device=self.device, dtype=self.dtype)
            u_pred = _jax_to_torch(u_jax).to(device=self.device, dtype=self.dtype)
            x_pred = x_pred[:, :, 1:].permute(0, 2, 1).contiguous()
            u_pred = u_pred.permute(0, 2, 1).contiguous()
            try:
                residual = float(res_jax)
            except Exception:
                residual = 0.0
            converged = torch.ones(B_batch, device=self.device, dtype=torch.bool)
            obj       = torch.full((B_batch,), residual, device=self.device, dtype=self.dtype)
            pimpc_solution = QPSolution(z=torch.empty(0), converged=converged,
                                        iters=0, obj=obj)
        except Exception as err:
            print(
                f"[LocoManipMPC] JAX PiMPC solve failed ({err}); "
                f"falling back to reference + warm-start u_prev."
            )
            x_pred = mpc_in.x_ref[:, 1:N + 1, :].contiguous()
            u_pred = mpc_in.u_prev.unsqueeze(1).expand(B_batch, N, nu).contiguous()
            pimpc_solution = self._make_fallback_solution(B_batch)

        t2 = self._tick() if do_profile else 0.0

        x_pred = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)
        u_pred = torch.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)

        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms": t1 - t0, "solve_ms": t2 - t1, "recover_ms": t3 - t2,
            }

        return self._make_output_manip(
            mpc_in, x_pred, u_pred, pimpc_solution,
        )

    def _make_fallback_solution(self, B: int) -> "QPSolution":
        """Stand-in :class:`QPSolution` used when PiMPC throws."""
        converged = torch.zeros(B, device=self.device, dtype=torch.bool)
        obj       = torch.zeros(B, device=self.device, dtype=self.dtype)
        return QPSolution(z=torch.empty(0), converged=converged, iters=0, obj=obj)

    def _solve_sparse_manip(self, mpc_in: LocoManipMPCInput) -> MPCOutput:
        """Sparse ADMM solve for the 18-D loco-manipulation QP."""
        cfg = self.cfg
        N   = cfg.N
        B_  = mpc_in.x0.shape[0]
        nx  = self._nx
        nu  = self._nu
        do_profile = cfg.profile

        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                 else mpc_in.x_ref[:, :N, :3])

        t0 = self._tick() if do_profile else 0.0

        Bk = self._build_Bk_manip(mpc_in)
        Bk = torch.nan_to_num(Bk, nan=0.0, posinf=0.0, neginf=0.0)
        qp = self._build_sparse_qp_manip(mpc_in, Bk)

        if self._warm_Z is None or self._warm_Z.shape[-1] != (nx + nu) * N:
            self._warm_Z = torch.cat([
                mpc_in.x_ref[:, 1:, :].reshape(B_, nx*N),
                mpc_in.u_ref.reshape(B_, nu*N),
            ], dim=-1)
            self._warm_lam_eq = None

        t1 = self._tick() if do_profile else 0.0

        sol = self.solver.solve(qp, warm_z=self._warm_Z,
                                warm_lam_eq=self._warm_lam_eq)
        Z_star = sol.z
        self._warm_Z      = Z_star.clone()
        self._warm_lam_eq = sol.lam_eq.clone() if sol.lam_eq is not None else None

        t2 = self._tick() if do_profile else 0.0

        x_pred = Z_star[:, :nx*N].reshape(B_, N, nx)
        u_pred = Z_star[:, nx*N:].reshape(B_, N, nu)

        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms":   t1 - t0,
                "solve_ms":   t2 - t1,
                "recover_ms": t3 - t2,
            }

        return self._make_output_manip(mpc_in, x_pred, u_pred, sol)

    def _make_output_manip(
        self,
        mpc_in: LocoManipMPCInput,
        x_pred: Tensor,
        u_pred: Tensor,
        sol,
    ) -> MPCOutput:
        """Pack the solve result into :class:`MPCOutput`."""
        N = self.cfg.N
        u_star  = u_pred[:, 0, :]
        sigma_0 = mpc_in.schedule.sigma[:, 0, :]
        x1      = x_pred[:, 0, :]
        x2      = x_pred[:, min(1, N - 1), :]
        z_mpc   = torch.cat([sigma_0, u_star, x1, x2], dim=-1)

        return MPCOutput(
            u_star=u_star,
            x_pred=x_pred,
            u_pred=u_pred,
            feasible=sol.converged,
            obj=sol.obj,
            z_mpc=z_mpc,
        )

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset warm-start buffers (same as parent but also handles 18-D size)."""
        if env_ids is None:
            self._warm_Z      = None
            self._warm_lam_eq = None
            self._warm_U      = None
        else:
            if self._warm_Z is not None:
                self._warm_Z[env_ids] = 0.0
            if self._warm_lam_eq is not None:
                self._warm_lam_eq[env_ids] = 0.0
            if self._warm_U is not None:
                self._warm_U[env_ids] = 0.0
