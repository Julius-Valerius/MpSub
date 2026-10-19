"""Deterministic high-dimensional verification and component ablation for MpSub.

Evaluates MpSub and its component ablations on a smooth deterministic
high-dimensional nonquadratic objective (quartic-augmented least squares,
n = 2000, m = 4000, effective rank r = 20) under a matched budget of
4,200 function evaluations across 5 random seeds.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def make_quartic_problem(n: int = 2000, r: int = 20, m: int = 4000,
                         seed: int = 42, iso: float = 0.1, qc: float = 0.05):
    """Construct a smooth deterministic nonquadratic objective in R^n with spectral decay."""
    rng = np.random.default_rng(seed)
    # Smooth power-law Hessian spectrum across all n dimensions, rotated by a dense Householder reflection
    idx = np.arange(1, n + 1, dtype=float)
    lam = 50.0 / (idx ** 0.75)  # eigenvalues decay smoothly from 50 to ~0.17 (condition number ~300)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    x_star = rng.standard_normal(n) / np.sqrt(n)

    def _rot(d: np.ndarray) -> np.ndarray:
        return d - 2.0 * float(v @ d) * v

    def f_eval(x: np.ndarray) -> float:
        z = _rot(x - x_star)
        quad = 0.5 * float(np.sum(lam * (z ** 2)))
        quart = qc * float(np.sum((lam * (z ** 2)) ** 2))
        return quad + quart

    def grad_norm(x: np.ndarray) -> float:
        z = _rot(x - x_star)
        gz = lam * z + 4.0 * qc * (lam ** 2) * (z ** 3)
        return float(np.linalg.norm(gz))

    x0 = np.zeros(n, dtype=float)
    return f_eval, grad_norm, x0


def run_mpsub_variant(
    f_eval,
    grad_norm,
    x0: np.ndarray,
    budget: int = 4200,
    p: int = 20,
    delta0: float = 0.1,
    eta: float = 0.1,
    gamma1: float = 0.5,
    gamma2: float = 2.0,
    delta_min: float = 1e-12,
    delta_max: float = 10.0,
    use_momentum: bool = True,
    accept_mode: str = "strict",   # "strict" (f+ < f) or "rho" (rho >= eta)
    adapt_radius: bool = True,     # True or False (fixed step size)
    kappa: float = 0.0,            # safeguard threshold ||g_k|| >= kappa * Delta_k
    seed: int = 0,
):
    """Run MpSub (Algorithm 1) or a single-component ablation under 2p+2 evals/step."""
    n = x0.size
    rng = np.random.default_rng(seed)
    x = x0.copy()
    m_vec = np.zeros(n, dtype=float)
    delta = float(delta0)

    evals = 0
    f0 = f_eval(x)
    g0 = grad_norm(x)

    accepted_steps = 0
    total_steps = 0
    expanded_steps = 0

    while evals + (2 * p + 2) <= budget:
        total_steps += 1
        fx = f_eval(x)
        evals += 1
        # 1. Subspace generation (exact Algorithm 1: non-orthogonal Gaussian frame)
        D = np.empty((p, n), dtype=float)
        m_norm = np.linalg.norm(m_vec)
        if use_momentum and m_norm > 0.0:
            D[0] = m_vec / m_norm
        else:
            D[0] = rng.standard_normal(n) / np.sqrt(n)
        D[1:] = rng.standard_normal((p - 1, n)) / np.sqrt(n)

        # 2. Central-difference subspace gradient estimation
        g = np.empty(p, dtype=float)
        for i in range(p):
            fp = f_eval(x + delta * D[i])
            fm = f_eval(x - delta * D[i])
            evals += 2
            g[i] = (fp - fm) / (2.0 * delta)

        gn = float(np.linalg.norm(g))
        if gn == 0.0:
            if adapt_radius:
                delta = max(gamma1 * delta, delta_min)
            continue

        # 3. Trust-region trial step
        s_s = -delta * (g / gn)
        u = s_s @ D  # D_k s_s in R^n
        x_plus = x + u
        f_plus = f_eval(x_plus)
        evals += 1

        pred = delta * gn
        rho = (fx - f_plus) / pred

        # 4. Acceptance update
        if accept_mode == "strict":
            accept = f_plus < fx
        elif accept_mode == "rho":
            accept = rho >= eta
        else:
            raise ValueError(f"Unknown accept_mode: {accept_mode}")

        if accept:
            m_vec = x_plus - x
            x = x_plus
            fx = f_plus
            accepted_steps += 1

        # 5. Radius update
        if adapt_radius:
            if rho >= eta and gn >= kappa * delta:
                delta = min(gamma2 * delta, delta_max)
                expanded_steps += 1
            else:
                delta = max(gamma1 * delta, delta_min)

    gn_final = grad_norm(x)
    return {
        "final_f": fx,
        "rel_f": fx / f0,
        "final_grad": gn_final,
        "rel_grad": gn_final / g0,
        "final_delta": delta,
        "accept_rate": accepted_steps / max(total_steps, 1),
        "expand_rate": expanded_steps / max(total_steps, 1),
        "steps": total_steps,
        "evals": evals,
    }


def main():
    f_eval, grad_norm, x0 = make_quartic_problem(n=1000, seed=42)
    f0 = f_eval(x0)
    g0 = grad_norm(x0)
    print(f"Problem initialized: n=1000, f(x0)={f0:.4f}, ||grad f(x0)||={g0:.4f}")

    seeds = [0, 1, 2, 3, 4]
    budget = 8400  # 200 iterations of MpSub at p=20 (matching main LLM budget)

    configs = [
        ("MpSub (default, p=20, Delta0=1e-1)", dict(p=20, delta0=0.1, use_momentum=True, accept_mode="strict", adapt_radius=True, kappa=0.0)),
        ("MpSub + safeguard (kappa=1e-3)", dict(p=20, delta0=0.1, use_momentum=True, accept_mode="strict", adapt_radius=True, kappa=1e-3)),
        ("No momentum (p=20 random)", dict(p=20, delta0=0.1, use_momentum=False, accept_mode="strict", adapt_radius=True, kappa=0.0)),
        ("Classical acceptance (rho >= eta)", dict(p=20, delta0=0.1, use_momentum=True, accept_mode="rho", adapt_radius=True, kappa=0.0)),
        ("MpSub (p=2 linear model)", dict(p=2, delta0=0.1, use_momentum=True, accept_mode="strict", adapt_radius=True, kappa=0.0)),
        ("MpSub from small Delta0=1e-3", dict(p=20, delta0=1e-3, use_momentum=True, accept_mode="strict", adapt_radius=True, kappa=0.0)),
        ("MpSub from large Delta0=1.0", dict(p=20, delta0=1.0, use_momentum=True, accept_mode="strict", adapt_radius=True, kappa=0.0)),
        ("Fixed radius (Delta_k = 1e-3)", dict(p=20, delta0=1e-3, use_momentum=True, accept_mode="strict", adapt_radius=False, kappa=0.0)),
        ("Fixed radius (Delta_k = 1.0)", dict(p=20, delta0=1.0, use_momentum=True, accept_mode="strict", adapt_radius=False, kappa=0.0)),
    ]

    summary = {}
    for label, kw in configs:
        runs = [run_mpsub_variant(f_eval, grad_norm, x0, budget=budget, seed=s, **kw) for s in seeds]
        mean_rel_f = float(np.mean([r["rel_f"] for r in runs]))
        min_rel_f = float(np.min([r["rel_f"] for r in runs]))
        max_rel_f = float(np.max([r["rel_f"] for r in runs]))
        mean_rel_g = float(np.mean([r["rel_grad"] for r in runs]))
        min_rel_g = float(np.min([r["rel_grad"] for r in runs]))
        max_rel_g = float(np.max([r["rel_grad"] for r in runs]))
        mean_acc = float(np.mean([r["accept_rate"] for r in runs]))
        mean_delta = float(np.mean([r["final_delta"] for r in runs]))
        summary[label] = {
            "rel_f_mean": mean_rel_f,
            "rel_f_min": min_rel_f,
            "rel_f_max": max_rel_f,
            "rel_grad_mean": mean_rel_g,
            "rel_grad_min": min_rel_g,
            "rel_grad_max": max_rel_g,
            "accept_rate_mean": mean_acc,
            "final_delta_mean": mean_delta,
            "steps": runs[0]["steps"],
            "evals": runs[0]["evals"],
        }
        print(
            f"{label:36s} | steps={runs[0]['steps']:4d} | "
            f"f/f0={mean_rel_f:.3f} ({min_rel_f:.3f}..{max_rel_f:.3f}) | "
            f"||g||/||g0||={mean_rel_g:.3f} ({min_rel_g:.3f}..{max_rel_g:.3f}) | "
            f"acc={mean_acc:.2f} | Delta_end={mean_delta:.2e}"
        )

    out_path = Path(__file__).resolve().parent / "ablation_deterministic_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
