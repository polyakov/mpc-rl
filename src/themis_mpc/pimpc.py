"""PiMPC - Parallel-in-Horizon MPC Solver (PyTorch)"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
import torch

@dataclass
class Results:
    """Solve results."""
    x: torch.Tensor
    u: torch.Tensor
    du: torch.Tensor
    solve_time: float
    iterations: int
    converged: bool
    obj_val: float

class Model:
    """PiMPC model. Create, configure with setup(), solve with solve()."""

    def __init__(self):
        self.is_setup = False
        self.warm_vars = None

    def setup(self,
              A, B, Np: int,
              C=None, e=None,
              Wy=None, Wu=None, Wdu=None, Wf=None,
              xmin=None, xmax=None,
              umin=None, umax=None,
              dumin=None, dumax=None,
              rho: float = 1.0,
              tol: float = 1e-4,
              eta: float = 0.999,
              maxiter: int = 100,
              precond: bool = False,
              accel: bool = False,
              device: str = "cpu",
              dtype=torch.float64):
        """Configure the MPC problem."""
        def _t(x, shape=None):
            """Convert array-like to tensor on device."""
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype)
            return torch.tensor(x, device=device, dtype=dtype)

        A = _t(A); B = _t(B)
        if B.dim() == 4:
            if B.shape[1] != Np:
                raise ValueError(
                    f"Per-env per-step B second dim ({B.shape[1]}) must equal Np ({Np})"
                )
            nx, nu = B.shape[2], B.shape[3]
        elif B.dim() == 3:
            if B.shape[0] != Np:
                raise ValueError(
                    f"Per-step B first dim ({B.shape[0]}) must equal Np ({Np})"
                )
            nx, nu = B.shape[1], B.shape[2]
        else:
            nx, nu = B.shape

        self.A = A
        self.B = B
        self.C = _t(C) if C is not None else torch.eye(nx, device=device, dtype=dtype)
        self.e = _t(e) if e is not None else torch.zeros(nx, device=device, dtype=dtype)

        ny = self.C.shape[0]
        self.nx, self.nu, self.ny, self.Np = nx, nu, ny, Np

        self.Wy  = _t(Wy)  if Wy  is not None else torch.eye(ny, device=device, dtype=dtype)
        self.Wu  = _t(Wu)  if Wu  is not None else torch.eye(nu, device=device, dtype=dtype)
        self.Wdu = _t(Wdu) if Wdu is not None else torch.eye(nu, device=device, dtype=dtype)
        self.Wf  = _t(Wf)  if Wf  is not None else self.Wy.clone()

        inf = float('inf')
        self.xmin = _t(xmin) if xmin is not None else torch.full((nx,), -inf, device=device, dtype=dtype)
        self.xmax = _t(xmax) if xmax is not None else torch.full((nx,),  inf, device=device, dtype=dtype)
        self.umin = _t(umin) if umin is not None else torch.full((nu,), -inf, device=device, dtype=dtype)
        self.umax = _t(umax) if umax is not None else torch.full((nu,),  inf, device=device, dtype=dtype)
        self.dumin = _t(dumin) if dumin is not None else torch.full((nu,), -inf, device=device, dtype=dtype)
        self.dumax = _t(dumax) if dumax is not None else torch.full((nu,),  inf, device=device, dtype=dtype)

        self.rho = rho
        self.tol = tol
        self.eta = eta
        self.maxiter = maxiter
        self.precond = precond
        self.accel = accel
        self.device = device
        self.dtype = dtype

        self.is_setup = True
        self.warm_vars = None

    def solve(self, x0, u0, yref, uref, w=None, verbose: bool = False) -> Results:
        """Solve MPC for a single initial condition."""
        assert self.is_setup, "Call setup() first."

        def _t(v):
            if v is None:
                return torch.zeros(self.nx, device=self.device, dtype=self.dtype)
            if isinstance(v, torch.Tensor):
                return v.to(device=self.device, dtype=self.dtype)
            return torch.tensor(v, device=self.device, dtype=self.dtype)

        x0   = _t(x0)
        u0   = _t(u0)
        yref = _t(yref)
        uref = _t(uref)
        w    = _t(w)

        x, u, du, info, warm = _solve(self, x0, u0, yref, uref, w,
                                      warm_vars=self.warm_vars,
                                      verbose=verbose)
        self.warm_vars = warm
        return Results(x=x, u=u, du=du, **info)

    def solve_batch(self, x0, u0, yref, uref, w=None,
                    umin_steps=None, umax_steps=None,
                    verbose: bool = False) -> Results:
        """Solve MPC for a batch of initial conditions simultaneously."""
        assert self.is_setup, "Call setup() first."

        def _t(v, default_shape):
            if v is None:
                return torch.zeros(default_shape, device=self.device, dtype=self.dtype)
            if isinstance(v, torch.Tensor):
                return v.to(device=self.device, dtype=self.dtype)
            return torch.tensor(v, device=self.device, dtype=self.dtype)

        x0   = _t(x0,   (1, self.nx))
        u0   = _t(u0,   (1, self.nu))
        yref = _t(yref, (1, self.ny))
        uref = _t(uref, (1, self.nu))
        w    = _t(w,    (1, self.nx))

        if x0.dim() == 1: x0 = x0.unsqueeze(0)
        batch = x0.shape[0]
        if u0.dim()   == 1: u0   = u0.unsqueeze(0).expand(batch, -1)
        if yref.dim() == 1: yref = yref.unsqueeze(0).expand(batch, -1)
        if uref.dim() == 1: uref = uref.unsqueeze(0).expand(batch, -1)
        if w.dim()    == 1: w    = w.unsqueeze(0).expand(batch, -1)

        if umin_steps is not None:
            umin_steps = umin_steps.to(device=self.device, dtype=self.dtype)
        if umax_steps is not None:
            umax_steps = umax_steps.to(device=self.device, dtype=self.dtype)

        x, u, du, info, _ = _solve_batch(self, x0, u0, yref, uref, w,
                                          umin_steps=umin_steps,
                                          umax_steps=umax_steps,
                                          verbose=verbose)
        return Results(x=x, u=u, du=du, **info)

def _solve(m: Model, x0, u0, yref, uref, w,
           warm_vars=None, verbose: bool = False):
    nx, nu, ny, Np = m.nx, m.nu, m.ny, m.Np
    nx_bar = nx + nu
    dev, dt = m.device, m.dtype
    rho, tol, max_iter, eta = m.rho, m.tol, m.maxiter, m.eta

    A_bar = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    A_bar[:nx, :nx] = m.A
    A_bar[:nx, nx:] = m.B
    A_bar[nx:, nx:] = torch.eye(nu, device=dev, dtype=dt)

    B_bar = torch.zeros(nx_bar, nu, device=dev, dtype=dt)
    B_bar[:nx, :] = m.B
    B_bar[nx:, :] = torch.eye(nu, device=dev, dtype=dt)

    C_bar = torch.zeros(ny, nx_bar, device=dev, dtype=dt)
    C_bar[:, :nx] = m.C

    e_bar = torch.cat([m.e, torch.zeros(nu, device=dev, dtype=dt)])
    w_bar = torch.cat([w,   torch.zeros(nu, device=dev, dtype=dt)])

    xmin_bar = torch.cat([m.xmin, m.umin])
    xmax_bar = torch.cat([m.xmax, m.umax])

    if m.precond:
        E = torch.sqrt((A_bar.T @ A_bar).diag())
        E_diag = torch.diag(E)
        E_inv  = torch.diag(1.0 / E)
        A_bar  = E_diag @ A_bar @ E_inv
        B_bar  = E_diag @ B_bar
        C_bar  = C_bar @ E_inv
        e_bar  = E_diag @ e_bar
        w_bar  = E_diag @ w_bar
        xmin_bar = E * xmin_bar
        xmax_bar = E * xmax_bar
    else:
        E_diag = torch.eye(nx_bar, device=dev, dtype=dt)
        E_inv  = torch.eye(nx_bar, device=dev, dtype=dt)

    C_part = C_bar[:, :nx]
    Q_bar   = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar[:nx, :nx] = C_part.T @ m.Wy @ C_part
    Q_bar[nx:, nx:] = m.Wu

    Q_bar_N = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar_N[:nx, :nx] = C_part.T @ m.Wf @ C_part
    Q_bar_N[nx:, nx:] = m.Wu

    q_bar   = torch.cat([C_part.T @ m.Wy @ yref, m.Wu @ uref])
    q_bar_N = torch.cat([C_part.T @ m.Wf @ yref, m.Wu @ uref])
    R_bar   = m.Wdu

    I_nb = torch.eye(nx_bar, device=dev, dtype=dt)
    J_B  = torch.linalg.solve(R_bar + rho * B_bar.T @ B_bar, B_bar.T)
    H_A  = torch.linalg.inv(Q_bar   + rho * I_nb + rho * A_bar.T @ A_bar)
    H_AN = torch.linalg.inv(Q_bar_N + rho * I_nb)

    x_bar = torch.cat([x0, u0])

    if warm_vars is None:
        DU     = torch.zeros(nu,    Np,    device=dev, dtype=dt)
        X      = torch.zeros(nx_bar, Np+1, device=dev, dtype=dt)
        X[:, 0] = E_diag @ x_bar
        V      = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Z      = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Theta  = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Beta   = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
        Lambda = torch.zeros(nx_bar, Np, device=dev, dtype=dt)
    else:
        DU0, X0, V0, Z0, Theta0, Beta0, Lambda0 = warm_vars
        DU     = torch.cat([DU0[:, 1:], DU0[:, -1:]], dim=1)
        X      = torch.cat([( E_diag @ x_bar).unsqueeze(1), X0[:, 2:], X0[:, -1:]], dim=1)
        V      = torch.cat([V0[:, 1:],      V0[:, -1:]],      dim=1)
        Z      = torch.cat([Z0[:, 1:],      Z0[:, -1:]],      dim=1)
        Theta  = torch.cat([Theta0[:, 1:],  Theta0[:, -1:]],  dim=1)
        Beta   = torch.cat([Beta0[:, 1:],   Beta0[:, -1:]],   dim=1)
        Lambda = torch.cat([Lambda0[:, 1:], Lambda0[:, -1:]], dim=1)
        X[:, 1:] = E_diag @ X[:, 1:]

    if m.accel:
        V_hat = V.clone();     Z_hat = Z.clone()
        Theta_hat = Theta.clone(); Beta_hat = Beta.clone(); Lambda_hat = Lambda.clone()

    Z_prev     = Z.clone();     V_prev     = V.clone()
    Theta_prev = Theta.clone(); Beta_prev  = Beta.clone(); Lambda_prev = Lambda.clone()

    residuals = []
    alpha_prev, res_prev, res = 1.0, float('inf'), float('inf')
    converged = False

    if verbose:
        print("PiMPC ADMM Solver")
        print("-" * 40)
        print(f"  {'Iter':>6}  {'Residual':>12}")
        print("-" * 40)

    t_start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Z_prev.copy_(Z);       V_prev.copy_(V)
        Theta_prev.copy_(Theta); Beta_prev.copy_(Beta); Lambda_prev.copy_(Lambda)

        if m.accel:
            _V, _Z, _Theta, _Beta, _Lambda = V_hat, Z_hat, Theta_hat, Beta_hat, Lambda_hat
        else:
            _V, _Z, _Theta, _Beta, _Lambda = V, Z, Theta, Beta, Lambda

        DU = J_B @ (_V - _Beta)

        if Np > 1:
            rhs_mid = (q_bar.unsqueeze(1)
                       + rho * (_Z[:, :Np-1] - _Theta[:, :Np-1]
                                + A_bar.T @ (_Z[:, 1:Np] - _V[:, 1:Np]
                                             + _Lambda[:, 1:Np]
                                             - e_bar.unsqueeze(1)
                                             - w_bar.unsqueeze(1))))
            X[:, 1:Np] = H_A @ rhs_mid

        X[:, Np] = H_AN @ (q_bar_N + rho * (_Z[:, Np-1] - _Theta[:, Np-1]))

        BU = B_bar @ DU
        AX = A_bar @ X[:, :Np]

        Z = (2.0 * (X[:, 1:Np+1] + _Theta) + BU + _Beta + AX + e_bar.unsqueeze(1) + w_bar.unsqueeze(1) - _Lambda) / 3.0
        Z = torch.clamp(Z, xmin_bar.unsqueeze(1), xmax_bar.unsqueeze(1))

        V = 0.5 * (Z + BU + _Beta - AX - e_bar.unsqueeze(1) - w_bar.unsqueeze(1) + _Lambda)

        Theta  = _Theta  + X[:, 1:Np+1] - Z
        Beta   = _Beta   + BU - V
        Lambda = _Lambda + Z - AX - V - e_bar.unsqueeze(1) - w_bar.unsqueeze(1)

        if m.accel:
            dT = Theta - _Theta;  dB = Beta - _Beta;  dL = Lambda - _Lambda
            dZ = Z - _Z;          dV = V - _V
        else:
            dT = Theta - Theta_prev;  dB = Beta - Beta_prev;  dL = Lambda - Lambda_prev
            dZ = Z - Z_prev;          dV = V - V_prev

        res = float(rho * (dT.norm()**2 + dB.norm()**2 + dL.norm()**2
                           + dZ.norm()**2 + dV.norm()**2
                           + ((dZ - dV)).norm()**2))
        residuals.append(res)

        if verbose:
            print(f"  {it:>6}  {res:>12.4e}")

        if res < tol:
            converged = True
            break

        if m.accel:
            if res < eta * res_prev:
                alpha = 0.5 * (1.0 + (1.0 + 4.0 * alpha_prev**2)**0.5)
                mom = (alpha_prev - 1.0) / alpha
                V_hat     = V     + mom * (V     - V_prev)
                Z_hat     = Z     + mom * (Z     - Z_prev)
                Theta_hat = Theta + mom * (Theta - Theta_prev)
                Beta_hat  = Beta  + mom * (Beta  - Beta_prev)
                Lambda_hat = Lambda + mom * (Lambda - Lambda_prev)
                res_prev = res
            else:
                alpha = 1.0
                V_hat.copy_(V); Z_hat.copy_(Z)
                Theta_hat.copy_(Theta); Beta_hat.copy_(Beta); Lambda_hat.copy_(Lambda)
                res_prev = res_prev / eta
            alpha_prev = alpha

    solve_time = time.perf_counter() - t_start

    if verbose:
        print("-" * 40)
        print(f"  Status:     {'Converged' if converged else 'Not converged'}")
        print(f"  Iterations: {len(residuals)}")
        print(f"  Time:       {solve_time*1000:.4f} ms")

    X = E_inv @ X
    x_traj = X[:nx, :]
    u_traj = X[nx:, 1:]

    info = dict(solve_time=solve_time,
                iterations=len(residuals),
                converged=converged,
                obj_val=residuals[-1] if residuals else float('inf'))
    warm = (DU, X, V, Z, Theta, Beta, Lambda)
    return x_traj, u_traj, DU, info, warm

def _solve_batch(m: Model, x0, u0, yref, uref, w,
                 umin_steps=None, umax_steps=None,
                 warm_vars=None, verbose: bool = False):
    """Batched ADMM.  All tensors have a leading batch dimension B."""
    nx, nu, ny, Np = m.nx, m.nu, m.ny, m.Np
    nx_bar = nx + nu
    dev, dt = m.device, m.dtype
    rho, tol, max_iter, eta = m.rho, m.tol, m.maxiter, m.eta
    B = x0.shape[0]

    per_env_step = (m.B.dim() == 4)
    per_step     = (m.B.dim() == 3)
    I_nu = torch.eye(nu, device=dev, dtype=dt)

    if per_env_step:
        n_env = m.B.shape[0]
        if n_env != B:
            raise ValueError(
                f"x0 batch size ({B}) must match m.B batch size ({n_env}) "
                f"when using per-env per-step dynamics."
            )
        A_bar_s = torch.zeros(B, Np, nx_bar, nx_bar, device=dev, dtype=dt)
        A_bar_s[:, :, :nx, :nx] = m.A.view(1, 1, nx, nx)
        A_bar_s[:, :, :nx, nx:] = m.B
        A_bar_s[:, :, nx:, nx:] = I_nu.view(1, 1, nu, nu)

        B_bar_s = torch.zeros(B, Np, nx_bar, nu, device=dev, dtype=dt)
        B_bar_s[:, :, :nx, :] = m.B
        B_bar_s[:, :, nx:, :] = I_nu.view(1, 1, nu, nu)

        A_bar = A_bar_s[0, 0]
        B_bar = B_bar_s[0, 0]
    elif per_step:
        A_bar_s = torch.zeros(Np, nx_bar, nx_bar, device=dev, dtype=dt)
        A_bar_s[:, :nx, :nx] = m.A.unsqueeze(0)
        A_bar_s[:, :nx, nx:] = m.B
        A_bar_s[:, nx:, nx:] = I_nu.unsqueeze(0)

        B_bar_s = torch.zeros(Np, nx_bar, nu, device=dev, dtype=dt)
        B_bar_s[:, :nx, :] = m.B
        B_bar_s[:, nx:, :] = I_nu.unsqueeze(0)

        A_bar = A_bar_s[0]
        B_bar = B_bar_s[0]
    else:
        A_bar = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
        A_bar[:nx, :nx] = m.A;  A_bar[:nx, nx:] = m.B
        A_bar[nx:, nx:] = I_nu

        B_bar = torch.zeros(nx_bar, nu, device=dev, dtype=dt)
        B_bar[:nx, :] = m.B
        B_bar[nx:, :] = I_nu

    C_bar = torch.zeros(ny, nx_bar, device=dev, dtype=dt)
    C_bar[:, :nx] = m.C

    e_bar = torch.cat([m.e, torch.zeros(nu, device=dev, dtype=dt)])
    w_bar = torch.cat([w.expand(B, -1), torch.zeros(B, nu, device=dev, dtype=dt)], dim=1)

    xmin_bar = torch.cat([m.xmin, m.umin])
    xmax_bar = torch.cat([m.xmax, m.umax])

    if umin_steps is not None:
        xmin_bar_steps = torch.empty(B, nx_bar, Np, device=dev, dtype=dt)
        xmax_bar_steps = torch.empty(B, nx_bar, Np, device=dev, dtype=dt)
        xmin_bar_steps[:, :nx, :] = m.xmin.view(1, nx, 1).expand(B, nx, Np)
        xmax_bar_steps[:, :nx, :] = m.xmax.view(1, nx, 1).expand(B, nx, Np)
        xmin_bar_steps[:, nx:, :] = umin_steps
        xmax_bar_steps[:, nx:, :] = umax_steps
    else:
        xmin_bar_steps = None
        xmax_bar_steps = None

    if m.precond:
        if per_step or per_env_step:
            raise NotImplementedError(
                "Preconditioning is not supported with per-step / per-env B."
            )
        E      = torch.sqrt((A_bar.T @ A_bar).diag())
        E_diag = torch.diag(E)
        E_inv  = torch.diag(1.0 / E)
        A_bar  = E_diag @ A_bar @ E_inv
        B_bar  = E_diag @ B_bar
        C_bar  = C_bar  @ E_inv
        e_bar  = E_diag @ e_bar
        w_bar  = (E_diag @ w_bar.T).T
        xmin_bar = E * xmin_bar
        xmax_bar = E * xmax_bar
    else:
        E_diag = torch.eye(nx_bar, device=dev, dtype=dt)
        E_inv  = torch.eye(nx_bar, device=dev, dtype=dt)

    C_part  = C_bar[:, :nx]
    Q_bar   = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar[:nx, :nx] = C_part.T @ m.Wy @ C_part
    Q_bar[nx:, nx:] = m.Wu

    Q_bar_N = torch.zeros(nx_bar, nx_bar, device=dev, dtype=dt)
    Q_bar_N[:nx, :nx] = C_part.T @ m.Wf @ C_part
    Q_bar_N[nx:, nx:] = m.Wu

    _CW  = C_part.T @ m.Wy
    _CWf = C_part.T @ m.Wf
    if yref.dim() == 3:
        q_y   = _CW @ yref
        q_u   = m.Wu @ uref
        q_bar = torch.cat([q_y, q_u], dim=1)
        q_bar_N = torch.cat([
            (_CWf @ yref[:, :, -1:]),
            (m.Wu @ uref[:, :, -1:]),
        ], dim=1)
    else:
        q_bar_const = torch.cat([
            (_CW @ yref.T).T,
            (m.Wu @ uref.T).T
        ], dim=1)
        q_bar = q_bar_const.unsqueeze(2).expand(B, nx_bar, Np).contiguous()
        q_bar_N = torch.cat([
            (_CWf @ yref.T).T,
            (m.Wu @ uref.T).T
        ], dim=1).unsqueeze(2)
    R_bar = m.Wdu

    I_nb = torch.eye(nx_bar, device=dev, dtype=dt)
    if per_env_step:
        BtB = B_bar_s.transpose(-1, -2) @ B_bar_s
        J_B_s = torch.linalg.solve(
            R_bar.view(1, 1, nu, nu) + rho * BtB,
            B_bar_s.transpose(-1, -2),
        )
        AtA = A_bar_s.transpose(-1, -2) @ A_bar_s
        H_A_s = torch.linalg.inv(
            Q_bar.view(1, 1, nx_bar, nx_bar)
            + rho * I_nb.view(1, 1, nx_bar, nx_bar)
            + rho * AtA
        )
    elif per_step:
        BtB = B_bar_s.transpose(-1, -2) @ B_bar_s
        J_B_s = torch.linalg.solve(
            R_bar.unsqueeze(0) + rho * BtB, B_bar_s.transpose(-1, -2)
        )
        AtA = A_bar_s.transpose(-1, -2) @ A_bar_s
        H_A_s = torch.linalg.inv(
            Q_bar.unsqueeze(0) + rho * I_nb.unsqueeze(0) + rho * AtA
        )
    else:
        J_B  = torch.linalg.solve(R_bar + rho * B_bar.T @ B_bar, B_bar.T)
        H_A  = torch.linalg.inv(Q_bar + rho * I_nb + rho * A_bar.T @ A_bar)
    H_AN = torch.linalg.inv(Q_bar_N + rho * I_nb)

    x_bar = torch.cat([x0, u0], dim=1)

    Ex_bar = (E_diag @ x_bar.T).T.unsqueeze(2)
    X      = torch.zeros(B, nx_bar, Np+1, device=dev, dtype=dt)
    X[:, :, :1] = Ex_bar
    DU     = torch.zeros(B, nu,    Np,   device=dev, dtype=dt)
    V      = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Z      = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Theta  = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Beta   = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)
    Lambda = torch.zeros(B, nx_bar, Np,  device=dev, dtype=dt)

    if m.accel:
        V_hat = V.clone(); Z_hat = Z.clone()
        Theta_hat = Theta.clone(); Beta_hat = Beta.clone(); Lambda_hat = Lambda.clone()

    Z_prev = Z.clone(); V_prev = V.clone()
    Theta_prev = Theta.clone(); Beta_prev = Beta.clone(); Lambda_prev = Lambda.clone()

    e_b = e_bar.view(1, nx_bar, 1)
    w_b = w_bar.unsqueeze(2)

    residuals = []
    alpha_prev, res_prev = 1.0, float('inf')
    converged = False

    if verbose:
        print(f"PiMPC ADMM Solver (batched B={B})")
        print("-" * 40)

    t_start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Z_prev.copy_(Z);        V_prev.copy_(V)
        Theta_prev.copy_(Theta); Beta_prev.copy_(Beta); Lambda_prev.copy_(Lambda)

        if m.accel:
            _V, _Z, _T, _Be, _La = V_hat, Z_hat, Theta_hat, Beta_hat, Lambda_hat
        else:
            _V, _Z, _T, _Be, _La = V, Z, Theta, Beta, Lambda

        if per_env_step:
            DU = torch.einsum('bkij,bjk->bik', J_B_s, _V - _Be)
        elif per_step:
            DU = torch.einsum('kij,bjk->bik', J_B_s, _V - _Be)
        else:
            DU = J_B @ (_V - _Be)

        if Np > 1:
            inner = (_Z[:, :, 1:Np] - _V[:, :, 1:Np] + _La[:, :, 1:Np]
                     - e_b.expand(B, -1, Np-1) - w_b.expand(B, -1, Np-1))
            if per_env_step:
                rhs_dyn = torch.einsum('bkji,bjk->bik', A_bar_s[:, 1:Np], inner)
                rhs_mid = (q_bar[:, :, :Np-1]
                           + rho * (_Z[:, :, :Np-1] - _T[:, :, :Np-1] + rhs_dyn))
                X[:, :, 1:Np] = torch.einsum('bkij,bjk->bik', H_A_s[:, 1:Np], rhs_mid)
            elif per_step:
                rhs_dyn = torch.einsum('kji,bjk->bik', A_bar_s[1:Np], inner)
                rhs_mid = (q_bar[:, :, :Np-1]
                           + rho * (_Z[:, :, :Np-1] - _T[:, :, :Np-1] + rhs_dyn))
                X[:, :, 1:Np] = torch.einsum('kij,bjk->bik', H_A_s[1:Np], rhs_mid)
            else:
                rhs_mid = (q_bar[:, :, :Np-1]
                           + rho * (_Z[:, :, :Np-1] - _T[:, :, :Np-1]
                                    + A_bar.T @ inner))
                X[:, :, 1:Np] = H_A @ rhs_mid

        rhs_N = q_bar_N + rho * (_Z[:, :, Np-1:Np] - _T[:, :, Np-1:Np])
        X[:, :, Np:Np+1] = H_AN @ rhs_N

        if per_env_step:
            BU = torch.einsum('bkij,bjk->bik', B_bar_s, DU)
            AX = torch.einsum('bkij,bjk->bik', A_bar_s, X[:, :, :Np])
        elif per_step:
            BU = torch.einsum('kij,bjk->bik', B_bar_s, DU)
            AX = torch.einsum('kij,bjk->bik', A_bar_s, X[:, :, :Np])
        else:
            BU = B_bar @ DU
            AX = A_bar @ X[:, :, :Np]

        ew = e_b.expand(B, -1, Np) + w_b.expand(B, -1, Np)

        Z  = (2.0 * (X[:, :, 1:Np+1] + _T) + BU + _Be + AX + ew - _La) / 3.0
        if xmin_bar_steps is not None:
            Z  = torch.clamp(Z, xmin_bar_steps, xmax_bar_steps)
        else:
            Z  = torch.clamp(Z, xmin_bar.view(1, -1, 1), xmax_bar.view(1, -1, 1))
        V  = 0.5 * (Z + BU + _Be - AX - ew + _La)

        Theta  = _T  + X[:, :, 1:Np+1] - Z
        Beta   = _Be + BU - V
        Lambda = _La + Z - AX - V - ew

        if m.accel:
            dT = Theta - _T;  dB = Beta - _Be;  dL = Lambda - _La
            dZ = Z - _Z;      dV = V - _V
        else:
            dT = Theta - Theta_prev; dB = Beta - Beta_prev; dL = Lambda - Lambda_prev
            dZ = Z - Z_prev;         dV = V - V_prev

        res = float(rho * (dT.norm()**2 + dB.norm()**2 + dL.norm()**2
                           + dZ.norm()**2 + dV.norm()**2
                           + (dZ - dV).norm()**2))
        residuals.append(res)

        if res < tol * B:
            converged = True
            break

        if m.accel:
            if res < eta * res_prev:
                alpha = 0.5 * (1.0 + (1.0 + 4.0 * alpha_prev**2)**0.5)
                mom = (alpha_prev - 1.0) / alpha
                V_hat     = V     + mom * (V     - V_prev)
                Z_hat     = Z     + mom * (Z     - Z_prev)
                Theta_hat = Theta + mom * (Theta - Theta_prev)
                Beta_hat  = Beta  + mom * (Beta  - Beta_prev)
                Lambda_hat = Lambda + mom * (Lambda - Lambda_prev)
                res_prev = res
            else:
                alpha = 1.0
                V_hat.copy_(V); Z_hat.copy_(Z)
                Theta_hat.copy_(Theta); Beta_hat.copy_(Beta); Lambda_hat.copy_(Lambda)
                res_prev = res_prev / eta
            alpha_prev = alpha

    solve_time = time.perf_counter() - t_start

    if verbose:
        print(f"  Status: {'Converged' if converged else 'Not converged'}, "
              f"iters={len(residuals)}, time={solve_time*1e3:.2f} ms")

    X = E_inv @ X
    x_traj = X[:, :nx, :]
    u_traj = X[:, nx:, 1:]

    info = dict(solve_time=solve_time,
                iterations=len(residuals),
                converged=converged,
                obj_val=residuals[-1] if residuals else float('inf'))
    return x_traj, u_traj, DU, info, None
