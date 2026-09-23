"""MpSub solver, CPU reference implementation.

p-dimensional subspace trust-region step with central-difference gradients.
The trust-region radius doubles as the finite-difference sampling radius and
persists across calls through PSubState, together with the momentum direction.

    state = PSubState(delta0=1e-1)
    x_best, f_best, state = psolve(f, x0, state, p=20)

Cost per step: 2p + 2 evaluations of f (one center evaluation, 2p central-difference values, and one trial value).
"""
import numpy as np


class PSubState:
    """State persisted across calls: radius delta and momentum direction d1."""

    def __init__(self, delta0=1e-1, dmax=None):
        self.delta = float(delta0)
        self.d1 = None
        self.dmax = float(dmax) if dmax is not None else 1.0
        self.dmin = 1e-12


def _build_directions(d1, p, n, rng, ortho):
    """B[0] = normalized d1; B[1:] = (optionally orthogonalized) Gaussians."""
    B = np.empty((p, n))
    B[0] = d1 / (np.linalg.norm(d1) + 1e-300)
    for i in range(1, p):
        v = rng.standard_normal(n)
        if ortho:
            v -= B[:i].T @ (B[:i] @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = rng.standard_normal(n)
            if ortho:
                v -= B[:i].T @ (B[:i] @ v)
            nv = np.linalg.norm(v)
        B[i] = v / (nv + 1e-300)
    return B


def psolve(f, x0, state, p=20, n_steps=1, eta=0.1,
           kind='grad', ortho=True, seed=0):
    """Run n_steps subspace trust-region iterations from x0.

    f        : callable(1D ndarray) -> float
    x0       : 1D ndarray, current parameters
    state    : PSubState, persisted across calls
    p        : subspace dimension
    eta      : acceptance threshold for the reduction ratio
    kind     : 'grad' (gradient-only step) or 'diag' (diagonal-curvature step)
    ortho    : Gram-Schmidt the random directions; unnecessary for large n
    seed     : vary per call so that directions differ across minibatches

    Returns (x_best, f_best, state).
    """
    rng = np.random.default_rng(seed)
    n = x0.size
    x = np.array(x0, dtype=float)
    fx = f(x)
    f_best, x_best = fx, x.copy()

    if state.d1 is None:
        state.d1 = np.zeros(n)
        state.d1[0] = 1.0

    for _ in range(n_steps):
        delta = state.delta
        B = _build_directions(state.d1, p, n, rng, ortho)

        fp = np.array([f(x + delta * B[i]) for i in range(p)])
        fm = np.array([f(x - delta * B[i]) for i in range(p)])
        g = (fp - fm) / (2.0 * delta)

        if kind == 'diag':
            hd = (fp + fm - 2.0 * fx) / (delta ** 2)
            s = np.where(hd > 1e-14, -g / np.maximum(hd, 1e-14),
                         -delta * np.sign(g))
            nrm = np.linalg.norm(s)
            if nrm > delta:
                s *= delta / nrm
            pred = float(g @ s + 0.5 * np.sum(hd * s * s))
        else:  # 'grad'
            gn = np.linalg.norm(g)
            s = -delta * g / max(gn, 1e-300)
            pred = float(g @ s)

        xn = x + B.T @ s
        fn = f(xn)
        rho = (fn - fx) / pred if (pred < -1e-14 and fn < fx) else -1.0

        if fn < fx:
            step_norm = np.linalg.norm(xn - x)
            if step_norm > 0:
                state.d1 = (xn - x) / step_norm
            x = xn
            fx = fn
            if fn < f_best:
                f_best = fn
                x_best = x.copy()

        state.delta = (min(delta * 2.0, state.dmax) if rho >= eta
                       else max(delta * 0.5, state.dmin))

    return x_best, f_best, state


def fp16_safety_report(delta, n, param_dtype_str, ref_w=0.02):
    """Check that the per-coordinate step delta/sqrt(n) is representable.

    A perturbation below half the floating-point spacing at the weight scale
    rounds to zero, in which case the finite-difference estimator is
    identically zero. Returns (safe, message).
    """
    per_param = float(delta) / max(np.sqrt(float(n)), 1.0)
    d = str(param_dtype_str).lower()
    if 'float16' in d or 'half' in d or 'bfloat16' in d:
        mant = 8 if 'bfloat' in d else 10
        exp = np.floor(np.log2(abs(ref_w))) if ref_w else -6.0
        spacing = 2.0 ** (exp - mant)
        safe = per_param > spacing / 2.0
        msg = ('[psub] per-param step %.3e vs %s spacing %.3e @w=%.3g -> %s'
               % (per_param, d, spacing, ref_w,
                  'OK' if safe else
                  'UNDERFLOW: increase psub_delta0 or use fp32'))
        return safe, msg
    return True, ('[psub] per-param step %.3e, dtype=%s' % (per_param, d))


def _dir_from_seed(base_seed, step_idx, i, n, dtype):
    """Deterministic unit direction from (seed, step, i).

    The same triple always yields the same vector, which allows the two-pass
    streaming variant to regenerate directions instead of storing B.
    """
    rng = np.random.default_rng([int(base_seed), int(step_idx), int(i)])
    v = rng.standard_normal(n, dtype=dtype)
    v /= (np.linalg.norm(v) + 1e-30)
    return v


def psolve_stream(f, x0, state, p=20, n_steps=1, eta=0.1,
                  seed=0, dtype=np.float32):
    """Streaming variant of psolve with O(n) peak memory instead of O(pn).

    Directions are regenerated from seeds in two passes (gradient pass, then
    step-accumulation pass) rather than stored; random directions are used
    without orthogonalization; the step is gradient-only. Evaluation count per
    step is 2p + 1, identical to psolve.
    """
    n = x0.size
    x = np.asarray(x0, dtype=dtype).copy()
    fx = f(x)
    f_best = float(fx)
    x_best = x.copy()

    if state.d1 is None or np.size(state.d1) != n:
        d1 = np.zeros(n, dtype=dtype)
        d1[0] = 1.0
        state.d1 = d1
    else:
        state.d1 = np.asarray(state.d1, dtype=dtype)

    buf = np.empty(n, dtype=dtype)   # reused probe buffer
    acc = np.empty(n, dtype=dtype)   # reused step accumulator

    for it in range(n_steps):
        delta = float(state.delta)

        # pass 1: subspace gradient by central differences
        g = np.empty(p, dtype=np.float64)
        for i in range(p):
            v = state.d1 if i == 0 else _dir_from_seed(seed, it, i, n, dtype)
            np.multiply(v, dtype(delta), out=buf)
            np.add(buf, x, out=buf)
            fpi = float(f(buf))
            np.multiply(v, dtype(-delta), out=buf)
            np.add(buf, x, out=buf)
            fmi = float(f(buf))
            g[i] = (fpi - fmi) / (2.0 * delta)

        gn = float(np.linalg.norm(g))
        if gn < 1e-300:
            state.delta = max(delta * 0.5, state.dmin)
            continue

        s = (-delta / gn) * g
        pred = float(g @ s)

        # pass 2: regenerate directions and accumulate the full-space step
        acc.fill(0)
        for i in range(p):
            if s[i] == 0.0:
                continue
            v = state.d1 if i == 0 else _dir_from_seed(seed, it, i, n, dtype)
            np.multiply(v, dtype(s[i]), out=buf)
            np.add(acc, buf, out=acc)

        np.add(x, acc, out=buf)
        fn = float(f(buf))
        rho = (fn - fx) / pred if (pred < -1e-14 and fn < fx) else -1.0

        if fn < fx:
            step_norm = float(np.linalg.norm(acc))
            if step_norm > 0:
                np.divide(acc, dtype(step_norm), out=state.d1)
            x[:] = buf
            fx = fn
            if fn < f_best:
                f_best = fn
                x_best[:] = x

        state.delta = (min(delta * 2.0, state.dmax) if rho >= eta
                       else max(delta * 0.5, state.dmin))

    return x_best, f_best, state
