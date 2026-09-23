"""Equal-budget surrogate study: subspace dimension and quadratic model.

Finite-sum least squares in n dimensions with effective Hessian rank r;
minibatch noise comes from genuine subsampling, and all evaluations within
one iteration share a batch. Compared at an equal evaluation budget with
per-method tuning: MeZO, gradient-only subspace steps at p in {2, 5, 20},
and the two-dimensional method with a full quadratic interpolation model.

Usage: python surrogate_dimension_model.py
       N=8000 BUD=8000 python surrogate_dimension_model.py
"""
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Stop(Exception):
    pass


def make_problem(n, r, m, seed=0, iso=0.05):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    A = rng.standard_normal((m, r)) @ U.T + iso * rng.standard_normal((m, n)) / np.sqrt(n)
    wstar = U @ rng.standard_normal(r)
    b = A @ wstar + 0.01 * rng.standard_normal(m)

    def f_full(w):
        res = A @ np.asarray(w, float) - b
        return float(0.5 * np.mean(res ** 2))

    def make_batch(bs, brng):
        if bs >= m:
            return f_full
        idx = brng.integers(0, m, bs)
        As, bb = A[idx].copy(), b[idx].copy()

        def fb(w):
            res = As @ np.asarray(w, float) - bb
            return float(0.5 * np.mean(res ** 2))
        return fb

    return f_full, make_batch, np.zeros(n)


def _tracker(f_full, f0, thresh=0.5):
    st = {'n': 0, 'best': np.inf, 'hit': None}

    def track(w):
        t = f_full(w)
        if t < st['best']:
            st['best'] = t
            if st['hit'] is None and t <= thresh * f0:
                st['hit'] = st['n']
    return st, track


def _guard(fb, st, budget):
    def F(w):
        if st['n'] >= budget:
            raise Stop
        st['n'] += 1
        return fb(w)
    return F


def mezo(f_full, make_batch, x0, f0, budget, bs, lr, eps=1e-2, seed=0):
    n = x0.size
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 8888)
    st, track = _tracker(f_full, f0)
    x = np.array(x0, float)
    track(x)
    try:
        while True:
            F = _guard(make_batch(bs, brng), st, budget)
            z = rng.standard_normal(n)
            d = (F(x + eps * z) - F(x - eps * z)) / (2 * eps)
            x = x - lr * d * z
            track(x)
    except Stop:
        pass
    return st['best'], st['hit']


def pgrad(f_full, make_batch, x0, f0, budget, bs, delta0, p, eta=0.1, seed=0):
    """Gradient-only p-dimensional subspace trust-region step; 2p+2 evals/step."""
    n = x0.size
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 8888)
    st, track = _tracker(f_full, f0)
    x = np.array(x0, float)
    track(x)
    d1 = np.zeros(n)
    d1[0] = 1.0
    delta, dmax = delta0, delta0 * 10
    try:
        while True:
            F = _guard(make_batch(bs, brng), st, budget)
            fx = F(x)
            B = np.empty((p, n))
            B[0] = d1 / np.linalg.norm(d1)
            for i in range(1, p):
                v = rng.standard_normal(n)
                v -= B[:i].T @ (B[:i] @ v)
                B[i] = v / np.linalg.norm(v)
            fp = np.array([F(x + delta * B[i]) for i in range(p)])
            fm = np.array([F(x - delta * B[i]) for i in range(p)])
            g = (fp - fm) / (2 * delta)
            gn = np.linalg.norm(g)
            if gn < 1e-300:
                continue
            s = -delta * g / gn
            xn = x + B.T @ s
            fn = F(xn)
            pred = float(g @ s)
            rho = (fn - fx) / pred if (pred < -1e-14 and fn < fx) else -1.0
            if fn < fx:
                step = np.linalg.norm(xn - x)
                if step > 0:
                    d1 = (xn - x) / step
                x = xn
                track(x)
            delta = min(delta * 2, dmax) if rho >= eta else max(delta * 0.5, 1e-14)
    except Stop:
        pass
    return st['best'], st['hit']


def mosub2d(f_full, make_batch, x0, f0, budget, bs, delta0, eta=0.1, seed=0):
    """Two-dimensional subspace with a full quadratic interpolation model and
    an exact trust-region subproblem solve; 7 evals/step."""
    n = x0.size
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 8888)
    st, track = _tracker(f_full, f0)
    x = np.array(x0, float)
    track(x)
    d1 = np.zeros(n)
    d1[0] = 1.0
    delta, dmax = delta0, delta0 * 10
    try:
        while True:
            F = _guard(make_batch(bs, brng), st, budget)
            fx = F(x)
            v = rng.standard_normal(n)
            v -= (v @ d1) * d1
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            B = np.array([d1, v / nv])
            fp = np.array([F(x + delta * B[i]) for i in range(2)])
            fm = np.array([F(x - delta * B[i]) for i in range(2)])
            g = (fp - fm) / (2 * delta)
            hd = (fp + fm - 2 * fx) / delta ** 2
            h = delta / np.sqrt(2.0)
            fij = F(x + h * (B[0] + B[1]))
            off = (fij - fx - h * (g[0] + g[1]) - 0.5 * h * h * (hd[0] + hd[1])) / (h * h)
            H = np.array([[hd[0], off], [off, hd[1]]])
            w, V = np.linalg.eigh(H)
            gh = V.T @ g
            s = None
            if w.min() > 1e-14:
                s_try = -(gh / w)
                if np.linalg.norm(s_try) <= delta:
                    s = s_try
            if s is None:   # boundary solution via bisection on the multiplier
                lo = max(0.0, -w.min()) + 1e-12
                hi = lo + max(1.0, np.linalg.norm(g) / max(delta, 1e-12)) + 1.0
                for _ in range(60):
                    if np.linalg.norm(gh / (w + hi)) > delta:
                        hi *= 2
                    else:
                        break
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if np.linalg.norm(gh / (w + mid)) > delta:
                        lo = mid
                    else:
                        hi = mid
                s = -(gh / (w + hi))
            s = V @ s
            xn = x + B.T @ s
            fn = F(xn)
            pred = float(g @ s + 0.5 * s @ H @ s)
            rho = (fn - fx) / pred if (pred < -1e-14 and fn < fx) else -1.0
            if fn < fx:
                step = np.linalg.norm(xn - x)
                if step > 0:
                    d1 = (xn - x) / step
                x = xn
                track(x)
            delta = min(delta * 2, dmax) if rho >= eta else max(delta * 0.5, 1e-14)
    except Stop:
        pass
    return st['best'], st['hit']


SEEDS = [0, 1, 2]
LRS = [10.0 ** k for k in range(-3, 3)]
DELTAS = [1e-2, 1e-1, 1.0]


def grid_best(runner, grid, f0):
    out = []
    for hp in grid:
        res = [runner(seed=s, **hp) for s in SEEDS]
        loss = float(np.mean([r[0] for r in res])) / f0
        hits = [r[1] for r in res if r[1] is not None]
        hit = int(np.mean(hits)) if len(hits) == len(res) else None
        out.append((loss, hit, hp))
    return min(out, key=lambda t: t[0])


if __name__ == '__main__':
    n = int(os.environ.get('N', 2000))
    budget = int(os.environ.get('BUD', 4000))
    r, m = 10, 4000
    f_full, mb, x0 = make_problem(n, r, m)
    f0 = f_full(x0)
    print('surrogate: n=%d r=%d m=%d budget=%d seeds=%d' %
          (n, r, m, budget, len(SEEDS)))
    print('metric: final full-sample loss normalized by f(x0); '
          'hit = evaluations to reach 0.5*f0')
    print()
    for bs in [64, 8]:
        print('--- batch=%d ---' % bs)
        rows = []
        v, hit, hp = grid_best(
            lambda seed=0, **hp: mezo(f_full, mb, x0, f0, budget, bs, seed=seed, **hp),
            [dict(lr=v) for v in LRS], f0)
        rows.append(('mezo', v, hit, hp))
        for p in [2, 5, 20]:
            v, hit, hp = grid_best(
                lambda seed=0, **hp: pgrad(f_full, mb, x0, f0, budget, bs, p=p,
                                           seed=seed, **hp),
                [dict(delta0=d) for d in DELTAS], f0)
            rows.append(('p%d_grad' % p, v, hit, hp))
        v, hit, hp = grid_best(
            lambda seed=0, **hp: mosub2d(f_full, mb, x0, f0, budget, bs,
                                         seed=seed, **hp),
            [dict(delta0=d) for d in DELTAS], f0)
        rows.append(('mosub2d_model', v, hit, hp))
        for name, v, hit, hp in rows:
            print('  %-16s loss=%-11.3e hit=%-7s %s' %
                  (name, v, hit if hit is not None else '-', hp), flush=True)
        print()
