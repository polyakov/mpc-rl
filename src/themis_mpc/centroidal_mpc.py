"""Contact-scheduled centroidal QP-MPC (bipedal locomotion)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import torch
from torch import Tensor

from themis_mpc.admm_qp import ADMMSolver, QPData, QPSolution
from themis_mpc.contact_schedule import ContactSchedule

@dataclass
class MPCConfig:
    """Centroidal MPC configuration (bipedal locomotion)."""

    N: int = 10
    dt: float = 0.07

    mass: float = 37.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    foot_x_toe:  float = 0.12
    foot_x_heel: float = 0.06
    foot_y_half: float = 0.04

    mu_foot: float = 0.6
    mu_foot_yaw: float = 0.1

    fz_max_foot: float = 800.0

    Q_c: tuple[float, ...] = (100.0, 100.0, 200.0)
    Q_l: tuple[float, ...] = (10.0, 10.0, 20.0)
    Q_k: tuple[float, ...] = (50.0, 50.0, 50.0)
    Qf_scale: float = 10.0

    R_f_foot: float = 1e-4
    R_tau_foot: float = 1e-4
    R_delta: float = 1e-3

    solver_type: str = "pimpc"
    admm_max_iter: int = 200
    admm_rho: float = 1.0
    admm_rho_eq: float = 1000.0
    admm_eps_abs: float = 1e-4
    admm_eps_rel: float = 1e-4
    pimpc_accel: bool = True
    pimpc_rho: float = 1.0
    pimpc_precondition: bool = True

    use_condensed: bool = False

    unconstrained: bool = False

    profile: bool = False

@dataclass
class MPCInput:
    """Inputs to the MPC at each control update.  Leading batch dim B on all tensors."""
    x0: Tensor
    schedule: ContactSchedule
    x_ref: Tensor
    u_ref: Tensor
    u_prev: Tensor
    c_bar: Tensor | None = None

@dataclass
class MPCOutput:
    """Outputs from the MPC solve."""
    u_star: Tensor
    x_pred: Tensor
    u_pred: Tensor
    feasible: Tensor
    obj: Tensor
    z_mpc: Tensor

_PiMPCModel = None
_JAX_PIMPC_MODULE = None

def _load_pimpc():
    """Lazily import the PyTorch PiMPC ``Model`` class."""
    global _PiMPCModel
    if _PiMPCModel is not None:
        return _PiMPCModel
    from themis_mpc.pimpc import Model
    _PiMPCModel = Model
    return Model

def _load_jax_pimpc():
    """Lazily import the JAX PiMPC solver (``themis_mpc.jax_pimpc``)."""
    global _JAX_PIMPC_MODULE
    if _JAX_PIMPC_MODULE is not None:
        return _JAX_PIMPC_MODULE
    from themis_mpc import jax_pimpc as _mod
    _JAX_PIMPC_MODULE = _mod
    return _mod

def _torch_to_jax(t: Tensor):
    """Torch → JAX, zero-copy via dlpack when possible."""
    import jax
    t = t.contiguous()
    try:
        return jax.dlpack.from_dlpack(t)
    except Exception:
        from torch.utils.dlpack import to_dlpack
        return jax.dlpack.from_dlpack(to_dlpack(t))

def _jax_to_torch(arr) -> Tensor:
    """JAX → Torch, zero-copy via dlpack when possible."""
    try:
        return torch.from_dlpack(arr)
    except Exception:
        import jax
        return torch.from_dlpack(jax.dlpack.to_dlpack(arr))

def _skew(v: Tensor) -> Tensor:
    """Batch skew-symmetric matrix [v]_x, shape (..., 3) → (..., 3, 3)."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)

def _make_T6(R: Tensor) -> Tensor:
    """Build block-diagonal rotation T6 = block_diag(R^T, R^T), shape (B, 6, 6)."""
    B = R.shape[0]
    RT = R.transpose(-1, -2)
    T  = torch.zeros(B, 6, 6, device=R.device, dtype=R.dtype)
    T[:, :3, :3] = RT
    T[:, 3:, 3:] = RT
    return T

class CentroidalMPC:
    """Contact-scheduled centroidal QP-MPC."""

    def __init__(self, cfg: MPCConfig, device: torch.device | str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)
        self.dtype  = torch.float32
        self.solver = ADMMSolver(
            max_iter=cfg.admm_max_iter,
            rho=cfg.admm_rho,
            rho_eq=cfg.admm_rho_eq,
            eps_abs=cfg.admm_eps_abs,
            eps_rel=cfg.admm_eps_rel,
        )
        self._build_constants()

        self._warm_Z: Tensor | None = None
        self._warm_lam_eq: Tensor | None = None

        self._warm_U: Tensor | None = None

        self.last_timing: dict[str, float] = {
            "setup_ms": 0.0, "solve_ms": 0.0, "recover_ms": 0.0
        }

        self._pimpc = None
        if cfg.solver_type == "pimpc":
            PiMPCModel = _load_pimpc()
            self._pimpc = PiMPCModel()

        self._jax_pimpc_solver = None
        self._jax_pimpc_shape: tuple[int, int] | None = None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _tick(self) -> float:
        """Return current wall-clock time in ms, with CUDA sync if needed."""
        self._sync()
        return time.perf_counter() * 1e3

    def _build_constants(self) -> None:
        """Precompute all time-invariant matrices (shared by both formulations)."""
        cfg = self.cfg
        dt, m = cfg.dt, cfg.mass
        g  = torch.tensor(cfg.gravity, device=self.device, dtype=self.dtype)
        I3 = torch.eye(3, device=self.device, dtype=self.dtype)
        nx, nu, N = 9, 12, cfg.N

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
        self.R       = torch.diag(R_diag)
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
        rows = [
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
        self._foot_cone_G = torch.tensor(rows, device=self.device, dtype=self.dtype)
        self._foot_cone_b = torch.zeros(12, device=self.device, dtype=self.dtype)
        self._foot_cone_b[5] = fz_max

    def _build_Bk(self, c_bar: Tensor, schedule: ContactSchedule) -> Tensor:
        """Per-step input matrix.  Returns (B, N, 9, 12)."""
        cfg = self.cfg
        B, N   = schedule.batch_size, schedule.horizon
        dt, m  = cfg.dt, cfg.mass
        device, dtype = self.device, self.dtype

        I3    = torch.eye(3, device=device, dtype=dtype)
        sigma = schedule.sigma
        r_LF  = schedule.r_LF
        r_RF  = schedule.r_RF

        Bk = torch.zeros(B, N, 9, 12, device=device, dtype=dtype)
        for k in range(N):
            c_k  = c_bar[:, k, :]
            s_LF = sigma[:, k, 0:1].unsqueeze(-1)
            s_RF = sigma[:, k, 1:2].unsqueeze(-1)

            Ef = torch.zeros(B, 3, 12, device=device, dtype=dtype)
            Ef[:, :, 0:3] = s_LF * I3.unsqueeze(0)
            Ef[:, :, 6:9] = s_RF * I3.unsqueeze(0)

            Et = torch.zeros(B, 3, 12, device=device, dtype=dtype)
            Et[:, :, 0:3]  = s_LF * _skew(r_LF[:, k, :] - c_k)
            Et[:, :, 3:6]  = s_LF * I3.unsqueeze(0)
            Et[:, :, 6:9]  = s_RF * _skew(r_RF[:, k, :] - c_k)
            Et[:, :, 9:12] = s_RF * I3.unsqueeze(0)

            Bk[:, k, 0:3, :] = (dt**2 / (2*m)) * Ef
            Bk[:, k, 3:6, :] = dt * Ef
            Bk[:, k, 6:9, :] = dt * Et

        return Bk

    def _build_prediction_matrices(
        self, Bk: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Condense dynamics: X = A_cal x0 + B_cal U + D_cal."""
        N  = self.cfg.N
        B  = Bk.shape[0]
        nx, nu = 9, 12
        device, dtype = self.device, self.dtype
        A, d = self.A, self.d

        A_cal = torch.zeros(B, nx*N, nx,   device=device, dtype=dtype)
        B_cal = torch.zeros(B, nx*N, nu*N, device=device, dtype=dtype)
        D_cal = torch.zeros(B, nx*N,       device=device, dtype=dtype)

        A_pow = A.unsqueeze(0).expand(B, -1, -1).clone()

        for i in range(N):
            rs, re = i*nx, (i+1)*nx
            A_cal[:, rs:re, :] = A_pow

            for j in range(i + 1):
                cs, ce = j*nu, (j+1)*nu
                if i == j:
                    B_cal[:, rs:re, cs:ce] = Bk[:, j, :, :]
                else:
                    A_ij = torch.matrix_power(A, i - j).unsqueeze(0)
                    B_cal[:, rs:re, cs:ce] = torch.bmm(
                        A_ij.expand(B, -1, -1), Bk[:, j, :, :]
                    )

            d_sum = torch.zeros(B, nx, device=device, dtype=dtype)
            for j in range(i + 1):
                A_ij = torch.matrix_power(A, i - j).unsqueeze(0).expand(B, -1, -1)
                d_sum += torch.bmm(
                    A_ij, d.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
                ).squeeze(-1)
            D_cal[:, rs:re] = d_sum

            A_pow = torch.bmm(A_pow, A.unsqueeze(0).expand(B, -1, -1))

        return A_cal, B_cal, D_cal

    def _build_condensed_qp(
        self,
        mpc_in: MPCInput,
        A_cal: Tensor,
        B_cal: Tensor,
        D_cal: Tensor,
    ) -> QPData:
        """Condensed QP: decision variable U = [u_0,...,u_{N-1}] ∈ R^{12N}."""
        cfg = self.cfg
        N  = cfg.N
        B  = mpc_in.x0.shape[0]
        nx, nu = 9, 12
        device, dtype = self.device, self.dtype

        X_ref   = mpc_in.x_ref[:, 1:, :].reshape(B, nx*N)
        U_ref   = mpc_in.u_ref.reshape(B, nu*N)
        X_const = (
            torch.bmm(A_cal, mpc_in.x0.unsqueeze(-1)).squeeze(-1) + D_cal - X_ref
        )

        Q_bar = self._Q_bar.unsqueeze(0).expand(B, -1, -1)

        BtQ = torch.bmm(B_cal.transpose(1, 2), Q_bar)
        H   = torch.bmm(BtQ, B_cal) + self._H_uu.unsqueeze(0).expand(B, -1, -1)
        H   = 0.5 * (H + H.transpose(1, 2))

        e_prev = torch.zeros(B, nu*N, device=device, dtype=dtype)
        e_prev[:, :nu] = -mpc_in.u_prev

        h = torch.bmm(BtQ, X_const.unsqueeze(-1)).squeeze(-1)
        h = h - torch.bmm(
            self._R_bar.unsqueeze(0).expand(B, -1, -1),
            U_ref.unsqueeze(-1),
        ).squeeze(-1)
        h = h + torch.bmm(
            self._DtRd.unsqueeze(0).expand(B, -1, -1),
            e_prev.unsqueeze(-1),
        ).squeeze(-1)

        if cfg.unconstrained:
            G = torch.zeros(B, 0, nu*N, device=device, dtype=dtype)
            b = torch.zeros(B, 0,       device=device, dtype=dtype)
        else:
            n_fc        = self._foot_cone_G.shape[0]
            n_ineq_step = 2 * n_fc
            n_ineq      = N * n_ineq_step

            G = torch.zeros(B, n_ineq, nu*N, device=device, dtype=dtype)
            b = torch.zeros(B, n_ineq,       device=device, dtype=dtype)

            fc_G_base = self._foot_cone_G
            fc_b      = self._foot_cone_b.unsqueeze(0).expand(B, -1)

            sch = mpc_in.schedule
            if sch.R_LF is not None and sch.R_RF is not None:
                T6_LF = _make_T6(sch.R_LF.to(device=device, dtype=dtype))
                T6_RF = _make_T6(sch.R_RF.to(device=device, dtype=dtype))
                fc_G_LF = torch.bmm(fc_G_base.unsqueeze(0).expand(B, -1, -1), T6_LF)
                fc_G_RF = torch.bmm(fc_G_base.unsqueeze(0).expand(B, -1, -1), T6_RF)
            else:
                fc_G_LF = fc_G_base.unsqueeze(0).expand(B, -1, -1)
                fc_G_RF = fc_G_base.unsqueeze(0).expand(B, -1, -1)

            for k in range(N):
                uo, co = k*nu, k*n_ineq_step
                G[:, co:co+n_fc,        uo:uo+6]    = fc_G_LF
                b[:, co:co+n_fc]                     = fc_b
                G[:, co+n_fc:co+2*n_fc, uo+6:uo+12] = fc_G_RF
                b[:, co+n_fc:co+2*n_fc]              = fc_b

        lb = torch.full((B, nu*N), -1e6, device=device, dtype=dtype)
        ub = torch.full((B, nu*N),  1e6, device=device, dtype=dtype)
        if not cfg.unconstrained:
            for k in range(N):
                uo      = k * nu
                sigma_k = mpc_in.schedule.sigma[:, k, :]
                inactive_LF = sigma_k[:, 0] < 0.5
                lb[inactive_LF, uo:uo+6]    = 0.0
                ub[inactive_LF, uo:uo+6]    = 0.0
                inactive_RF = sigma_k[:, 1] < 0.5
                lb[inactive_RF, uo+6:uo+12] = 0.0
                ub[inactive_RF, uo+6:uo+12] = 0.0

        return QPData(H=H, h=h, G=G, b=b, lb=lb, ub=ub)

    def _build_sparse_qp(self, mpc_in: MPCInput, Bk: Tensor) -> QPData:
        """Sparse QP: Z = [x_1,...,x_N, u_0,...,u_{N-1}] ∈ R^{21N}."""
        cfg = self.cfg
        N  = cfg.N
        B  = mpc_in.x0.shape[0]
        nx, nu = 9, 12
        n_z = (nx + nu) * N
        xs, us = 0, nx * N
        device, dtype = self.device, self.dtype

        H = self._H_full.unsqueeze(0).expand(B, -1, -1)

        X_ref  = mpc_in.x_ref[:, 1:, :].reshape(B, nx*N)
        U_ref  = mpc_in.u_ref.reshape(B, nu*N)
        e_prev = torch.zeros(B, nu*N, device=device, dtype=dtype)
        e_prev[:, :nu] = mpc_in.u_prev

        h = torch.zeros(B, n_z, device=device, dtype=dtype)
        h[:, xs:xs+nx*N] = -(self._Q_bar.unsqueeze(0) @ X_ref.unsqueeze(-1)).squeeze(-1)
        h[:, us:us+nu*N] = (
            -(self._R_bar.unsqueeze(0)  @ U_ref.unsqueeze(-1)).squeeze(-1)
            -(self._DtRd.unsqueeze(0)   @ e_prev.unsqueeze(-1)).squeeze(-1)
        )

        n_eq = nx * N
        A_eq = torch.zeros(B, n_eq, n_z, device=device, dtype=dtype)
        b_eq = torch.zeros(B, n_eq,      device=device, dtype=dtype)
        I_nx  = torch.eye(nx, device=device, dtype=dtype)
        d_B   = self.d.unsqueeze(0).expand(B, -1)
        rhs_0 = mpc_in.x0 @ self.A.t() + self.d

        for k in range(N):
            rs, re = k*nx, (k+1)*nx
            A_eq[:, rs:re, xs+k*nx : xs+(k+1)*nx] = I_nx
            if k > 0:
                A_eq[:, rs:re, xs+(k-1)*nx : xs+k*nx] = -self.A
            A_eq[:, rs:re, us+k*nu : us+(k+1)*nu] = -Bk[:, k, :, :]
            b_eq[:, rs:re] = rhs_0 if k == 0 else d_B

        if cfg.unconstrained:
            G = torch.zeros(B, 0, n_z, device=device, dtype=dtype)
            b = torch.zeros(B, 0,      device=device, dtype=dtype)
        else:
            n_fc        = self._foot_cone_G.shape[0]
            n_ineq_step = 2 * n_fc
            n_ineq      = N * n_ineq_step
            G = torch.zeros(B, n_ineq, n_z, device=device, dtype=dtype)
            b = torch.zeros(B, n_ineq,      device=device, dtype=dtype)

            fc_G_base = self._foot_cone_G
            fc_b      = self._foot_cone_b.unsqueeze(0).expand(B, -1)

            sch = mpc_in.schedule
            if sch.R_LF is not None and sch.R_RF is not None:
                T6_LF = _make_T6(sch.R_LF.to(device=device, dtype=dtype))
                T6_RF = _make_T6(sch.R_RF.to(device=device, dtype=dtype))
                fc_G_LF = torch.bmm(fc_G_base.unsqueeze(0).expand(B, -1, -1), T6_LF)
                fc_G_RF = torch.bmm(fc_G_base.unsqueeze(0).expand(B, -1, -1), T6_RF)
            else:
                fc_G_LF = fc_G_base.unsqueeze(0).expand(B, -1, -1)
                fc_G_RF = fc_G_base.unsqueeze(0).expand(B, -1, -1)

            for k in range(N):
                uo, co = us + k*nu, k*n_ineq_step
                G[:, co:co+n_fc,        uo:uo+6]    = fc_G_LF
                b[:, co:co+n_fc]                     = fc_b
                G[:, co+n_fc:co+2*n_fc, uo+6:uo+12] = fc_G_RF
                b[:, co+n_fc:co+2*n_fc]              = fc_b

        lb = torch.full((B, n_z), -1e6, device=device, dtype=dtype)
        ub = torch.full((B, n_z),  1e6, device=device, dtype=dtype)
        if not cfg.unconstrained:
            for k in range(N):
                uo      = us + k * nu
                sigma_k = mpc_in.schedule.sigma[:, k, :]
                inactive_LF = sigma_k[:, 0] < 0.5
                lb[inactive_LF, uo:uo+6]    = 0.0
                ub[inactive_LF, uo:uo+6]    = 0.0
                inactive_RF = sigma_k[:, 1] < 0.5
                lb[inactive_RF, uo+6:uo+12] = 0.0
                ub[inactive_RF, uo+6:uo+12] = 0.0

        return QPData(H=H, h=h, G=G, b=b, lb=lb, ub=ub, A_eq=A_eq, b_eq=b_eq)

    def solve(self, mpc_in: MPCInput) -> MPCOutput:
        """Solve centroidal MPC.  Dispatches based on solver_type and formulation."""
        st = self.cfg.solver_type
        if st == "pimpc":
            return self._solve_pimpc(mpc_in)
        elif st == "jax_pimpc":
            return self._solve_jax_pimpc(mpc_in)
        elif st == "admm":
            if self.cfg.use_condensed:
                return self._solve_condensed(mpc_in)
            else:
                return self._solve_sparse(mpc_in)
        else:
            raise ValueError(
                f"Unknown MPCConfig.solver_type: {st!r}. "
                f"Expected one of: 'admm', 'pimpc', 'jax_pimpc'."
            )

    def _solve_pimpc(self, mpc_in: MPCInput) -> MPCOutput:
        """Solve via PiMPC parallel-in-horizon ADMM (per-env, per-step B_k)."""
        cfg = self.cfg
        N = cfg.N
        B_batch = mpc_in.x0.shape[0]
        nx, nu = 9, 12
        do_profile = cfg.profile

        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                 else mpc_in.x_ref[:, :N, :3])

        t0 = self._tick() if do_profile else 0.0

        if cfg.unconstrained:
            sched_for_B = ContactSchedule(
                sigma=torch.ones_like(mpc_in.schedule.sigma),
                r_LF=mpc_in.schedule.r_LF, r_RF=mpc_in.schedule.r_RF,
                r_LH=mpc_in.schedule.r_LH, r_RH=mpc_in.schedule.r_RH,
                R_LF=mpc_in.schedule.R_LF, R_RF=mpc_in.schedule.R_RF,
            )
        else:
            sched_for_B = mpc_in.schedule
        Bk = self._build_Bk(c_bar, sched_for_B)

        if cfg.unconstrained:
            umin = torch.full((nu,), -1e6, device=self.device, dtype=self.dtype)
            umax = torch.full((nu,),  1e6, device=self.device, dtype=self.dtype)
        else:
            mu    = cfg.mu_foot
            fz_max = cfg.fz_max_foot
            yh    = cfg.foot_y_half
            toe   = cfg.foot_x_toe
            heel  = cfg.foot_x_heel
            mu_z  = cfg.mu_foot_yaw
            umin = torch.tensor([
                -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
                -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            ], device=self.device, dtype=self.dtype)
            umax = torch.tensor([
                mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
                mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            ], device=self.device, dtype=self.dtype)

        yref = mpc_in.x_ref[:, 1:N+1, :].permute(0, 2, 1).contiguous()
        uref = mpc_in.u_ref.permute(0, 2, 1).contiguous()
        x0_in = mpc_in.x0
        u0_in = mpc_in.u_prev

        umin_steps = umin.view(1, nu, 1).expand(B_batch, nu, N).clone()
        umax_steps = umax.view(1, nu, 1).expand(B_batch, nu, N).clone()
        if not cfg.unconstrained:
            sigma   = mpc_in.schedule.sigma
            lf_act  = sigma[:, :, 0:1].permute(0, 2, 1).float()
            rf_act  = sigma[:, :, 1:2].permute(0, 2, 1).float()
            umin_steps[:, :6, :] *= lf_act
            umax_steps[:, :6, :] *= lf_act
            umin_steps[:, 6:, :] *= rf_act
            umax_steps[:, 6:, :] *= rf_act

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

        result = self._pimpc.solve_batch(x0_in, u0_in, yref, uref,
                                         umin_steps=umin_st,
                                         umax_steps=umax_st)

        t2 = self._tick() if do_profile else 0.0

        x_pred = result.x[:, :, 1:].permute(0, 2, 1).contiguous()
        u_pred = result.u.permute(0, 2, 1).contiguous()

        if cfg.pimpc_precondition:
            s = self._pimpc_ruiz_cache
            x_pred = x_pred * s["dx"].view(1, 1, nx)
            u_pred = u_pred * s["du"].view(1, 1, nu)

        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms": t1 - t0, "solve_ms": t2 - t1, "recover_ms": t3 - t2,
            }

        return self._make_output(mpc_in, x_pred, u_pred,
                                 self._make_pimpc_solution(result, B_batch))

    def _make_pimpc_solution(self, result, B: int) -> "QPSolution":
        """Wrap PiMPC Results into a QPSolution-compatible object for _make_output."""
        converged = torch.full((B,), result.converged, device=self.device, dtype=torch.bool)
        obj = torch.full((B,), result.obj_val, device=self.device, dtype=self.dtype)
        return QPSolution(z=torch.empty(0), converged=converged, iters=result.iterations, obj=obj)

    def _get_pimpc_ruiz_cache(self, nu: int) -> dict:
        """Lazily build the Ruiz-scaling factors and pre-scaled invariant"""
        cache = getattr(self, "_pimpc_ruiz_cache", None)
        if cache is not None:
            return cache
        Wdu_unscaled = self.cfg.R_delta * torch.eye(
            nu, device=self.device, dtype=self.dtype,
        )
        dx  = 1.0 / torch.sqrt(torch.diag(self.Q))
        du  = 1.0 / torch.sqrt(torch.diag(self.R))
        Dx  = torch.diag(dx)
        Dxi = torch.diag(1.0 / dx)
        Du  = torch.diag(du)
        Dui = torch.diag(1.0 / du)
        cache = {
            "dx":  dx,  "du":  du,
            "Dx":  Dx,  "Dxi": Dxi,
            "Du":  Du,  "Dui": Dui,
            "A_s":   Dxi @ self.A  @ Dx,
            "e_s":   Dxi @ self.d,
            "Wy_s":  Dx  @ self.Q  @ Dx,
            "Wf_s":  Dx  @ self.Qf @ Dx,
            "Wu_s":  Du  @ self.R  @ Du,
            "Wdu_s": Du  @ Wdu_unscaled @ Du,
        }
        self._pimpc_ruiz_cache = cache
        return cache

    def _solve_jax_pimpc(self, mpc_in: MPCInput) -> MPCOutput:
        """Solve via the JAX PiMPC port (``themis_mpc.jax_pimpc``)."""
        cfg = self.cfg
        N = cfg.N
        B_batch = mpc_in.x0.shape[0]
        nx, nu = 9, 12
        do_profile = cfg.profile

        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                 else mpc_in.x_ref[:, :N, :3])

        t0 = self._tick() if do_profile else 0.0

        if cfg.unconstrained:
            sched_for_B = ContactSchedule(
                sigma=torch.ones_like(mpc_in.schedule.sigma),
                r_LF=mpc_in.schedule.r_LF, r_RF=mpc_in.schedule.r_RF,
                r_LH=mpc_in.schedule.r_LH, r_RH=mpc_in.schedule.r_RH,
                R_LF=mpc_in.schedule.R_LF, R_RF=mpc_in.schedule.R_RF,
            )
        else:
            sched_for_B = mpc_in.schedule
        Bk = self._build_Bk(c_bar, sched_for_B)

        if cfg.unconstrained:
            umin = torch.full((nu,), -1e6, device=self.device, dtype=self.dtype)
            umax = torch.full((nu,),  1e6, device=self.device, dtype=self.dtype)
        else:
            mu, fz_max = cfg.mu_foot, cfg.fz_max_foot
            yh, toe, heel = cfg.foot_y_half, cfg.foot_x_toe, cfg.foot_x_heel
            mu_z = cfg.mu_foot_yaw
            umin = torch.tensor([
                -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
                -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            ], device=self.device, dtype=self.dtype)
            umax = torch.tensor([
                mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
                mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            ], device=self.device, dtype=self.dtype)

        umin_steps = umin.view(1, nu, 1).expand(B_batch, nu, N).clone()
        umax_steps = umax.view(1, nu, 1).expand(B_batch, nu, N).clone()
        if not cfg.unconstrained:
            sigma   = mpc_in.schedule.sigma
            lf_act  = sigma[:, :, 0:1].permute(0, 2, 1).float()
            rf_act  = sigma[:, :, 1:2].permute(0, 2, 1).float()
            umin_steps[:, :6, :] *= lf_act
            umax_steps[:, :6, :] *= lf_act
            umin_steps[:, 6:, :] *= rf_act
            umax_steps[:, 6:, :] *= rf_act

        yref = mpc_in.x_ref[:, 1:N+1, :].permute(0, 2, 1).contiguous()
        uref = mpc_in.u_ref.permute(0, 2, 1).contiguous()

        Wdu = cfg.R_delta * torch.eye(nu, device=self.device, dtype=self.dtype)

        import jax.numpy as jnp
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

        e_be = self.d.unsqueeze(0).expand(B_batch, nx).contiguous()
        prob = {
            "A":          _torch_to_jax(self.A),
            "B_s":        _torch_to_jax(Bk),
            "e":          _torch_to_jax(e_be),
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

        t2 = self._tick() if do_profile else 0.0
        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms": t1 - t0, "solve_ms": t2 - t1, "recover_ms": t3 - t2,
            }

        try:
            residual = float(res_jax)
        except Exception:
            residual = 0.0
        converged = torch.ones(B_batch, device=self.device, dtype=torch.bool)
        obj       = torch.full((B_batch,), residual, device=self.device, dtype=self.dtype)
        sol       = QPSolution(z=torch.empty(0), converged=converged, iters=0, obj=obj)
        return self._make_output(mpc_in, x_pred, u_pred, sol)

    def _solve_sparse(self, mpc_in: MPCInput) -> MPCOutput:
        """Sparse joint (x, u) solve.  States extracted by direct slice."""
        cfg = self.cfg
        N  = cfg.N
        B  = mpc_in.x0.shape[0]
        nx, nu = 9, 12
        do_profile = cfg.profile

        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                 else mpc_in.x_ref[:, :N, :3])

        t0 = self._tick() if do_profile else 0.0

        Bk = self._build_Bk(c_bar, mpc_in.schedule)
        qp = self._build_sparse_qp(mpc_in, Bk)

        if self._warm_Z is None:
            self._warm_Z = torch.cat([
                mpc_in.x_ref[:, 1:, :].reshape(B, nx*N),
                mpc_in.u_ref.reshape(B, nu*N),
            ], dim=-1)

        t1 = self._tick() if do_profile else 0.0

        sol = self.solver.solve(qp, warm_z=self._warm_Z,
                                warm_lam_eq=self._warm_lam_eq)
        Z_star = sol.z
        self._warm_Z      = Z_star.clone()
        self._warm_lam_eq = sol.lam_eq.clone() if sol.lam_eq is not None else None

        t2 = self._tick() if do_profile else 0.0

        x_pred = Z_star[:, :nx*N].reshape(B, N, nx)
        u_pred = Z_star[:, nx*N:].reshape(B, N, nu)

        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms":   t1 - t0,
                "solve_ms":   t2 - t1,
                "recover_ms": t3 - t2,
            }

        return self._make_output(mpc_in, x_pred, u_pred, sol)

    def _solve_condensed(self, mpc_in: MPCInput) -> MPCOutput:
        """Condensed U-only solve.  States recovered via A_cal x0 + B_cal U + D_cal."""
        cfg = self.cfg
        N  = cfg.N
        B  = mpc_in.x0.shape[0]
        nx, nu = 9, 12
        do_profile = cfg.profile

        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                 else mpc_in.x_ref[:, :N, :3])

        t0 = self._tick() if do_profile else 0.0

        Bk                  = self._build_Bk(c_bar, mpc_in.schedule)
        A_cal, B_cal, D_cal = self._build_prediction_matrices(Bk)
        qp                  = self._build_condensed_qp(mpc_in, A_cal, B_cal, D_cal)

        t1 = self._tick() if do_profile else 0.0

        sol    = self.solver.solve(qp, warm_z=self._warm_U)
        U_star = sol.z
        self._warm_U = U_star.clone()

        t2 = self._tick() if do_profile else 0.0

        X_star = (
            torch.bmm(A_cal, mpc_in.x0.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(B_cal, U_star.unsqueeze(-1)).squeeze(-1)
            + D_cal
        )
        x_pred = X_star.reshape(B, N, nx)
        u_pred = U_star.reshape(B, N, nu)

        t3 = self._tick() if do_profile else 0.0

        if do_profile:
            self.last_timing = {
                "setup_ms":   t1 - t0,
                "solve_ms":   t2 - t1,
                "recover_ms": t3 - t2,
            }

        return self._make_output(mpc_in, x_pred, u_pred, sol)

    def _make_output(
        self,
        mpc_in: MPCInput,
        x_pred: Tensor,
        u_pred: Tensor,
        sol,
    ) -> MPCOutput:
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
        """Reset warm start.  Pass env_ids to reset specific envs, None to reset all."""
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
