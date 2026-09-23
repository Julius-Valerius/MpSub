"""MpSub solver, GPU in-place implementation.

Same algorithm as psub_solver.psolve with two implementation changes for
large models: parameters are perturbed in place on their device (no flatten,
no host copies), and directions are regenerated from seeds with the device
RNG instead of being stored.  The persistent accepted displacement and the
trial-step accumulator are FP32 tensors of model size, so auxiliary memory is
O(n) and independent of p.  For Gaussian z, ||z|| ~ sqrt(n) up to
O(1/sqrt(n)) relative error, so z/sqrt(n) is used as a unit direction without
a norm pass.

    state = TorchPSubState(delta0=1e-1)
    loss, state = psolve_torch(f_eval, params, state, p=20)

f_eval : callable() -> float, loss at the current parameter values
params : list[torch.Tensor], trainable parameters, modified in place
With n_steps=1, cost per training step is 2p + 2 evaluations of f_eval: one
centre value, 2p central-difference values, and one trial value.
"""
import math


class TorchPSubState:
    """State persisted across steps: radius delta and momentum direction."""

    def __init__(self, delta0=1e-1, dmax=None):
        self.delta = float(delta0)
        self.dmax = float(dmax) if dmax is not None else 1.0
        self.dmin = 1e-12
        self.mom = None          # list[Tensor] matching params
        self.mom_norm = 0.0


def _seed_for(base_seed, step_idx, i):
    """Deterministic seed for direction i of step step_idx."""
    return (int(base_seed) * 1000003 + int(step_idx) * 10007 + int(i) * 101) % (2 ** 31 - 1)


def _add_dir(params, state, gen, seed_val, i, alpha, inv_sqrt_n, accum=None):
    """params += alpha * v_i, optionally accumulating alpha * v_i into accum.

    v_0 is the normalized momentum direction when available; otherwise
    v_i = z / sqrt(n) regenerated from seed_val.
    """
    import torch
    if i == 0 and state.mom is not None and state.mom_norm > 0:
        scale = alpha / state.mom_norm
        for idx, (prm, m) in enumerate(zip(params, state.mom)):
            prm.data.add_(m, alpha=scale)
            if accum is not None:
                accum[idx].add_(m, alpha=scale)
        return
    gen.manual_seed(int(seed_val))
    a = alpha * inv_sqrt_n
    for idx, prm in enumerate(params):
        z = torch.randn(prm.shape, generator=gen, device=prm.device, dtype=torch.float32)
        prm.data.add_(z.to(prm.dtype), alpha=a)
        if accum is not None:
            accum[idx].add_(z, alpha=a)


def psolve_torch(f_eval, params, state, p=20, n_steps=1, eta=0.1, seed=0):
    """Run n_steps subspace trust-region iterations in place on params.

    Returns (best_loss, state). Vary seed per call so that directions differ
    across minibatches.
    """
    import torch
    dev = params[0].device
    gen = torch.Generator(device=dev)
    n = sum(prm.numel() for prm in params)
    inv_sqrt_n = 1.0 / math.sqrt(n)

    fx = float(f_eval())
    f_best = fx

    for it in range(n_steps):
        delta = float(state.delta)
        g = [0.0] * p

        # pass 1: subspace gradient by central differences
        for i in range(p):
            sv = _seed_for(seed, it, i)
            _add_dir(params, state, gen, sv, i, +delta, inv_sqrt_n)
            fp = float(f_eval())
            _add_dir(params, state, gen, sv, i, -2.0 * delta, inv_sqrt_n)
            fm = float(f_eval())
            _add_dir(params, state, gen, sv, i, +delta, inv_sqrt_n)  # restore
            g[i] = (fp - fm) / (2.0 * delta)

        gn = math.sqrt(sum(v * v for v in g))
        if gn < 1e-30:
            state.delta = max(delta * 0.5, state.dmin)
            continue

        s = [-delta * v / gn for v in g]
        pred = sum(gi * si for gi, si in zip(g, s))

        # pass 2: regenerate directions, move to the trial point, record step
        accum = [torch.zeros(prm.shape, device=prm.device, dtype=torch.float32)
                 for prm in params]
        for i in range(p):
            if s[i] == 0.0:
                continue
            _add_dir(params, state, gen, _seed_for(seed, it, i), i,
                     s[i], inv_sqrt_n, accum=accum)

        fn = float(f_eval())
        rho = (fn - fx) / pred if (pred < -1e-14 and fn < fx) else -1.0

        if fn < fx:
            state.mom = accum
            state.mom_norm = math.sqrt(sum(float(a.pow(2).sum()) for a in accum))
            fx = fn
            f_best = min(f_best, fn)
        else:
            for prm, a in zip(params, accum):
                prm.data.sub_(a.to(prm.dtype))
            del accum

        state.delta = (min(delta * 2.0, state.dmax) if rho >= eta
                       else max(delta * 0.5, state.dmin))

    return f_best, state
