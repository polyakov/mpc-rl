"""PiMPC — Parallel-in-Horizon MPC solver, ported to JAX."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

def _build_augmented(A, B_s, Wy, Wu, Wf, nx, nu):
    """Augmented per-(env,step) system + cost matrices."""
    Bn, Np = B_s.shape[0], B_s.shape[1]
    nb = nx + nu
    I_nu = jnp.eye(nu)

    A_bar_s = jnp.zeros((Bn, Np, nb, nb))
    A_bar_s = A_bar_s.at[:, :, :nx, :nx].set(A.reshape(1, 1, nx, nx))
    A_bar_s = A_bar_s.at[:, :, :nx, nx:].set(B_s)
    A_bar_s = A_bar_s.at[:, :, nx:, nx:].set(I_nu.reshape(1, 1, nu, nu))

    B_bar_s = jnp.zeros((Bn, Np, nb, nu))
    B_bar_s = B_bar_s.at[:, :, :nx, :].set(B_s)
    B_bar_s = B_bar_s.at[:, :, nx:, :].set(I_nu.reshape(1, 1, nu, nu))

    Q_bar = jnp.zeros((nb, nb)).at[:nx, :nx].set(Wy).at[nx:, nx:].set(Wu)
    Q_bar_N = jnp.zeros((nb, nb)).at[:nx, :nx].set(Wf).at[nx:, nx:].set(Wu)
    return A_bar_s, B_bar_s, Q_bar, Q_bar_N

@partial(jax.jit, static_argnames=("maxiter", "accel"))
def solve_batch(A, B_s, e, Wy, Wu, Wdu, Wf,
                x0, u0, yref, uref, umin_steps, umax_steps,
                rho=1.0, maxiter=200, eta=0.999, accel=True):
    """Batched PiMPC ADMM solve (per-env per-step dynamics)."""
    B, Np = B_s.shape[0], B_s.shape[1]
    nx, nu = A.shape[0], B_s.shape[3]
    nb = nx + nu

    A_bar_s, B_bar_s, Q_bar, Q_bar_N = _build_augmented(A, B_s, Wy, Wu, Wf, nx, nu)

    # Per-env affine/disturbance term e: [B, nx] -> e_b: [B, nb, 1] (broadcasts
    # over the horizon).  A single global disturbance is passed as [B, nx] with
    # identical rows.
    e_bar = jnp.concatenate([e, jnp.zeros((B, nu))], axis=1)
    e_b = e_bar.reshape(B, nb, 1)

    big = 1e9
    xmin = jnp.concatenate([jnp.full((B, nx, Np), -big), umin_steps], axis=1)
    xmax = jnp.concatenate([jnp.full((B, nx, Np), big), umax_steps], axis=1)

    q_y = jnp.einsum("ij,bjk->bik", Wy, yref)
    q_u = jnp.einsum("ij,bjk->bik", Wu, uref)
    q_bar = jnp.concatenate([q_y, q_u], axis=1)
    q_bar_N = jnp.concatenate([
        jnp.einsum("ij,bjk->bik", Wf, yref[:, :, -1:]),
        jnp.einsum("ij,bjk->bik", Wu, uref[:, :, -1:]),
    ], axis=1)

    I_nb = jnp.eye(nb)
    BtB = jnp.einsum("bkji,bkjl->bkil", B_bar_s, B_bar_s)
    J_B_s = jnp.linalg.solve(Wdu.reshape(1, 1, nu, nu) + rho * BtB,
                             jnp.swapaxes(B_bar_s, -1, -2))
    AtA = jnp.einsum("bkji,bkjl->bkil", A_bar_s, A_bar_s)
    H_A_s = jnp.linalg.inv(Q_bar.reshape(1, 1, nb, nb)
                           + rho * I_nb.reshape(1, 1, nb, nb) + rho * AtA)
    H_AN = jnp.linalg.inv(Q_bar_N + rho * I_nb)

    x_bar = jnp.concatenate([x0, u0], axis=1)
    X0col = x_bar.reshape(B, nb, 1)

    zero_v = jnp.zeros((B, nb, Np))
    V = zero_v; Z = zero_v; T = zero_v; Be = zero_v; La = zero_v
    Vh, Zh, Th, Beh, Lah = V, Z, T, Be, La

    def admm_step(_V, _Z, _T, _Be, _La):
        DU = jnp.einsum("bkij,bjk->bik", J_B_s, _V - _Be)
        inner = (_Z[:, :, 1:Np] - _V[:, :, 1:Np] + _La[:, :, 1:Np]
                 - e_b)
        rhs_dyn = jnp.einsum("bkji,bjk->bik", A_bar_s[:, 1:Np], inner)
        rhs_mid = q_bar[:, :, :Np-1] + rho * (_Z[:, :, :Np-1]
                                              - _T[:, :, :Np-1] + rhs_dyn)
        X_mid = jnp.einsum("bkij,bjk->bik", H_A_s[:, 1:Np], rhs_mid)
        rhs_N = q_bar_N + rho * (_Z[:, :, Np-1:Np] - _T[:, :, Np-1:Np])
        X_term = jnp.einsum("ij,bjk->bik", H_AN, rhs_N)
        X = jnp.concatenate([X0col, X_mid, X_term], axis=2)

        BU = jnp.einsum("bkij,bjk->bik", B_bar_s, DU)
        AX = jnp.einsum("bkij,bjk->bik", A_bar_s, X[:, :, :Np])
        Z_new = (2.0 * (X[:, :, 1:Np+1] + _T) + BU + _Be + AX + e_b - _La) / 3.0
        Z_new = jnp.clip(Z_new, xmin, xmax)
        V_new = 0.5 * (Z_new + BU + _Be - AX - e_b + _La)
        T_new = _T + X[:, :, 1:Np+1] - Z_new
        Be_new = _Be + BU - V_new
        La_new = _La + Z_new - AX - V_new - e_b
        return X, DU, Z_new, V_new, T_new, Be_new, La_new

    def residual(dT, dB, dL, dZ, dV):
        n2 = lambda a: jnp.sum(a * a)
        return rho * (n2(dT) + n2(dB) + n2(dL) + n2(dZ) + n2(dV) + n2(dZ - dV))

    init = (V, Z, T, Be, La, Vh, Zh, Th, Beh, Lah,
            jnp.array(1.0), jnp.array(jnp.inf), jnp.array(jnp.inf),
            jnp.broadcast_to(X0col, (B, nb, Np + 1)), jnp.zeros((B, nu, Np)))

    def body(_, c):
        (V, Z, T, Be, La, Vh, Zh, Th, Beh, Lah,
         alpha_prev, res_prev, _res, _X, _DU) = c
        if accel:
            _V, _Z, _T, _Be, _La = Vh, Zh, Th, Beh, Lah
        else:
            _V, _Z, _T, _Be, _La = V, Z, T, Be, La

        X, DU, Zn, Vn, Tn, Ben, Lan = admm_step(_V, _Z, _T, _Be, _La)

        if accel:
            dT, dB, dL = Tn - _T, Ben - _Be, Lan - _La
            dZ, dV = Zn - _Z, Vn - _V
        else:
            dT, dB, dL = Tn - T, Ben - Be, Lan - La
            dZ, dV = Zn - Z, Vn - V
        res = residual(dT, dB, dL, dZ, dV)

        if accel:
            do = res < eta * res_prev
            alpha = jnp.where(do, 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * alpha_prev**2)), 1.0)
            mom = jnp.where(do, (alpha_prev - 1.0) / alpha, 0.0)
            Vh2 = jnp.where(do, Vn + mom * (Vn - V), Vn)
            Zh2 = jnp.where(do, Zn + mom * (Zn - Z), Zn)
            Th2 = jnp.where(do, Tn + mom * (Tn - T), Tn)
            Beh2 = jnp.where(do, Ben + mom * (Ben - Be), Ben)
            Lah2 = jnp.where(do, Lan + mom * (Lan - La), Lan)
            res_prev2 = jnp.where(do, res, res_prev / eta)
            alpha_prev2 = alpha
        else:
            Vh2, Zh2, Th2, Beh2, Lah2 = Vn, Zn, Tn, Ben, Lan
            res_prev2, alpha_prev2 = res_prev, alpha_prev

        return (Vn, Zn, Tn, Ben, Lan, Vh2, Zh2, Th2, Beh2, Lah2,
                alpha_prev2, res_prev2, res, X, DU)

    out = lax.fori_loop(0, maxiter, body, init)
    X, res = out[13], out[12]
    x_traj = X[:, :nx, :]
    u_traj = X[:, nx:, 1:]
    return x_traj, u_traj, res

_ARG_KEYS = ("A", "B_s", "e", "Wy", "Wu", "Wdu", "Wf",
             "x0", "u0", "yref", "uref", "umin_steps", "umax_steps")

class PiMPCSolver:
    """Ahead-of-time-compiled PiMPC, mirroring the mpx wrapper pattern."""

    def __init__(self, B, N, nx=9, nu=12, *, maxiter=200, accel=True,
                 precondition=False, dtype=jnp.float32, compile_now=True):
        self.B, self.N, self.nx, self.nu = B, N, nx, nu
        self.maxiter, self.accel = maxiter, accel
        self.precondition, self.dtype = precondition, dtype

        def _run(A, B_s, e, Wy, Wu, Wdu, Wf, x0, u0, yref, uref, umn, umx, rho):
            if precondition:
                dx = 1.0 / jnp.sqrt(jnp.diag(Wy))
                du = 1.0 / jnp.sqrt(jnp.diag(Wu))
                Dx, Dxi = jnp.diag(dx), jnp.diag(1.0 / dx)
                Du, Dui = jnp.diag(du), jnp.diag(1.0 / du)
                A = Dxi @ A @ Dx
                B_s = jnp.einsum("ij,bkjl,lm->bkim", Dxi, B_s, Du)
                e = e @ Dxi
                Wy, Wf = Dx @ Wy @ Dx, Dx @ Wf @ Dx
                Wu, Wdu = Du @ Wu @ Du, Du @ Wdu @ Du
                x0, u0 = x0 @ Dxi, u0 @ Dui
                yref = jnp.einsum("ij,bjk->bik", Dxi, yref)
                uref = jnp.einsum("ij,bjk->bik", Dui, uref)
                umn = jnp.einsum("ij,bjk->bik", Dui, umn)
                umx = jnp.einsum("ij,bjk->bik", Dui, umx)
                x, u, r = solve_batch(A, B_s, e, Wy, Wu, Wdu, Wf, x0, u0, yref,
                                      uref, umn, umx, rho=rho,
                                      maxiter=maxiter, accel=accel)
                return x * dx[None, :, None], u * du[None, :, None], r
            return solve_batch(A, B_s, e, Wy, Wu, Wdu, Wf, x0, u0, yref, uref,
                               umn, umx, rho=rho, maxiter=maxiter, accel=accel)

        self._jit = jax.jit(_run)
        self._compiled = None
        if compile_now:
            self.compile()

    def compile(self):
        """AOT-compile the solve for the configured shapes/dtype (no data needed)."""
        s = lambda *shp: jax.ShapeDtypeStruct(shp, self.dtype)
        B, N, nx, nu = self.B, self.N, self.nx, self.nu
        self._compiled = self._jit.lower(
            s(nx, nx), s(B, N, nx, nu), s(B, nx), s(nx, nx), s(nu, nu), s(nu, nu),
            s(nx, nx), s(B, nx), s(B, nu), s(B, nx, N), s(B, nu, N),
            s(B, nu, N), s(B, nu, N), s(),
        ).compile()
        return self

    def solve(self, prob, rho=1.0):
        """Run the precompiled solver on a problem dict; returns (x, u, residual)."""
        fn = self._compiled if self._compiled is not None else self._jit
        args = [jnp.asarray(prob[k], self.dtype) for k in _ARG_KEYS]
        return fn(*args, jnp.asarray(rho, self.dtype))
