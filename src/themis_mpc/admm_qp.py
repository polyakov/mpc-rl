"""GPU-batched ADMM QP solver in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass
class QPData:
    """Batched QP data. All tensors have leading batch dim B."""

    H:     Tensor
    h:     Tensor
    G:     Tensor
    b:     Tensor
    lb:    Tensor
    ub:    Tensor

    A_eq:  Tensor | None = None
    b_eq:  Tensor | None = None

@dataclass
class QPSolution:
    """Solution returned by the ADMM solver."""

    z: Tensor
    converged: Tensor
    iters: int
    obj: Tensor
    lam_eq: Tensor | None = None

class ADMMSolver:
    """GPU-batched ADMM QP solver."""

    def __init__(
        self,
        max_iter: int = 200,
        rho: float = 1.0,
        rho_eq: float = 100.0,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        adaptive_rho: bool = True,
        adaptive_rho_interval: int = 25,
    ):
        self.max_iter = max_iter
        self.rho = rho
        self.rho_eq = rho_eq
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_interval = adaptive_rho_interval

    @torch.no_grad()
    def solve(
        self,
        qp: QPData,
        warm_z: Tensor | None = None,
        warm_lam_eq: Tensor | None = None,
    ) -> QPSolution:
        """Solve a batch of QPs."""
        B, n = qp.h.shape
        m    = qp.b.shape[1]
        device = qp.h.device
        dtype  = qp.h.dtype
        has_eq = qp.A_eq is not None

        rho = self.rho

        z = (warm_z.clone() if warm_z is not None
             else torch.zeros(B, n, device=device, dtype=dtype))
        Gz = torch.bmm(qp.G, z.unsqueeze(-1)).squeeze(-1)
        y  = torch.minimum(Gz, qp.b)
        u  = torch.zeros(B, m, device=device, dtype=dtype)

        z_box = z.clone()
        u_box = torch.zeros(B, n, device=device, dtype=dtype)

        if has_eq:
            n_eq   = qp.b_eq.shape[1]
            lam_eq = (warm_lam_eq.clone() if warm_lam_eq is not None
                      else torch.zeros(B, n_eq, device=device, dtype=dtype))
            rho_eq = self.rho_eq
            AtA_eq = torch.bmm(qp.A_eq.transpose(1, 2), qp.A_eq)
        else:
            AtA_eq = None

        GtG = torch.bmm(qp.G.transpose(1, 2), qp.G)
        I_n = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
        rho_t = torch.full((B, 1, 1), rho, device=device, dtype=dtype)

        def _factorize(rho_val: Tensor) -> tuple:
            M = qp.H + rho_val * (GtG + I_n)
            if has_eq:
                M = M + rho_eq * AtA_eq
            return torch.linalg.lu_factor(M)

        LU, pivots = _factorize(rho_t)

        if has_eq:
            Atb_eq = torch.bmm(
                qp.A_eq.transpose(1, 2), qp.b_eq.unsqueeze(-1)
            ).squeeze(-1)

        iters_used = 0
        r_ineq = torch.zeros(B, m, device=device, dtype=dtype)
        r_box  = torch.zeros(B, n,  device=device, dtype=dtype)

        for it in range(self.max_iter):
            rhs = -qp.h + rho_t.squeeze(-1) * (
                torch.bmm(qp.G.transpose(1, 2), (y - u).unsqueeze(-1)).squeeze(-1)
                + (z_box - u_box)
            )
            if has_eq:
                rhs = rhs + rho_eq * Atb_eq - torch.bmm(
                    qp.A_eq.transpose(1, 2), lam_eq.unsqueeze(-1)
                ).squeeze(-1)
            z = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(-1)).squeeze(-1)

            Gz = torch.bmm(qp.G, z.unsqueeze(-1)).squeeze(-1)
            y  = torch.minimum(Gz + u, qp.b)

            z_box = torch.clamp(z + u_box, min=qp.lb, max=qp.ub)

            r_ineq = Gz - y
            r_box  = z - z_box
            u     = u     + r_ineq
            u_box = u_box + r_box

            if has_eq:
                eq_res = torch.bmm(qp.A_eq, z.unsqueeze(-1)).squeeze(-1) - qp.b_eq
                lam_eq = lam_eq + rho_eq * eq_res

            iters_used = it + 1

            if (it + 1) % 10 == 0:
                primal_res = torch.sqrt(
                    (r_ineq * r_ineq).sum(-1) + (r_box * r_box).sum(-1)
                )
                if has_eq:
                    primal_res = primal_res + (eq_res * eq_res).sum(-1).sqrt()

                dual_scale = rho * torch.sqrt(
                    (u * u).sum(-1) + (u_box * u_box).sum(-1)
                )

                eps_pri = (
                    self.eps_abs * (m + n) ** 0.5
                    + self.eps_rel * torch.maximum(
                        torch.sqrt((Gz * Gz).sum(-1) + (z * z).sum(-1)),
                        torch.sqrt((y * y).sum(-1) + (z_box * z_box).sum(-1)),
                    )
                )
                eps_dual = self.eps_abs * n**0.5 + self.eps_rel * dual_scale

                converged = (primal_res < eps_pri) & (dual_scale < eps_dual)
                if converged.all():
                    break

            if self.adaptive_rho and (it + 1) % self.adaptive_rho_interval == 0:
                primal_norm = torch.sqrt(
                    (r_ineq * r_ineq).sum(-1) + (r_box * r_box).sum(-1)
                ).mean()
                dual_norm = dual_scale.mean()
                rho_changed = False
                if primal_norm > 10.0 * dual_norm and rho < 1e3:
                    rho = min(rho * 2.0, 1e3)
                    rho_changed = True
                    u = u / 2.0; u_box = u_box / 2.0
                elif dual_norm > 10.0 * primal_norm and rho > 1e-3:
                    rho = max(rho / 2.0, 1e-3)
                    rho_changed = True
                    u = u * 2.0; u_box = u_box * 2.0
                if rho_changed:
                    rho_t = torch.full((B, 1, 1), rho, device=device, dtype=dtype)
                    LU, pivots = _factorize(rho_t)

        z_final = torch.clamp(z, min=qp.lb, max=qp.ub)

        obj = (
            0.5 * (z_final * torch.bmm(qp.H, z_final.unsqueeze(-1)).squeeze(-1)).sum(-1)
            + (qp.h * z_final).sum(-1)
        )

        Gz_final = torch.bmm(qp.G, z_final.unsqueeze(-1)).squeeze(-1)
        ineq_ok  = (Gz_final <= qp.b + self.eps_abs).all(dim=-1)
        box_ok   = (
            (z_final >= qp.lb - self.eps_abs) & (z_final <= qp.ub + self.eps_abs)
        ).all(dim=-1)
        if has_eq:
            eq_res_final = torch.bmm(qp.A_eq, z_final.unsqueeze(-1)).squeeze(-1) - qp.b_eq
            eq_ok = (eq_res_final.abs() <= self.eps_abs * 100).all(dim=-1)
        else:
            eq_ok = torch.ones(B, device=device, dtype=torch.bool)

        return QPSolution(
            z=z_final,
            converged=ineq_ok & box_ok & eq_ok,
            iters=iters_used,
            obj=obj,
            lam_eq=lam_eq if has_eq else None,
        )
