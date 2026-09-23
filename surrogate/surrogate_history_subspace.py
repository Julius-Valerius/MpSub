"""Equal-budget surrogate study: basis construction from past gradient estimates.

Finite-sum least squares in n dimensions with effective Hessian rank r and
genuine minibatch subsampling. All four subspace variants share the same p,
the same gradient-only trust-region step and the same evaluation budget; the
only difference is which directions occupy the leading rows of the basis:

    nomom   B[0:] all random; no memory of the previous step (control arm)
    rand    B[0] = last accepted displacement, B[1:] random
    gradmom B[0] = displacement, B[1] = EMA of past gradient estimates
    hist    B[0] = displacement, B[1:1+nhist] = the last nhist estimates
    oracle  B[0] = displacement, B[1] = exact full-sample gradient

The nomom arm is the control for the momentum direction itself: it is
identical to rand in every respect except that the leading row of the basis
is a fresh random direction rather than the previous accepted displacement,
so the rand/nomom ratio isolates the value of the memory in the subspace.
The oracle direction is not charged to the budget and is not implementable;
it bounds what any basis-quality improvement can achieve. MeZO is included
to locate the gap.

Usage: python surrogate_history_subspace.py
       N=8000 BUD=8000 P=20 python surrogate_history_subspace.py
"""
import os
from collections import deque
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

    def g_full(w):
        return A.T @ (A @ np.asarray(w, float) - b) / m

    def make_batch(bs, brng):
        if bs >= m:
            return f_full
        idx = brng.integers(0, m, bs)
        As, bb = A[idx].copy(), b[idx].copy()

        def fb(w):
            res = As @ np.asarray(w, float) - bb
            return float(0.5 * np.mean(res ** 2))

        return fb

    return f_full, g_full, make_batch, np.zeros(n)


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


def _build_basis(n, p, specials, rng):
    """Place specials first by Gram-Schmidt, then fill with random directions."""
    B = np.zeros((p, n))
    k = 0
    for v in specials:
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        u = v / nv
        u = u - B[:k].T @ (B[:k] @ u)
        nv = np.linalg.norm(u)
        if nv < 1e-8:           # nearly parallel to an accepted direction
            continue
        B[k] = u / nv
        k += 1
        if k == p:
            return B
    while k < p:
        v = rng.standard_normal(n)
        v -= B[:k].T @ (B[:k] @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        B[k] = v / nv
        k += 1
    return B


def psub(f_full, g_full, make_batch, x0, f0, budget, bs, delta0,
         mode='rand', p=20, beta=0.9, nhist=4, eta=0.1, seed=0):
    n = x0.size
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 8888)
    st, track = _tracker(f_full, f0)
    x = np.array(x0, float)
    track(x)
    d1 = np.zeros(n)
    d1[0] = 1.0
    mom = np.zeros(n)
    hist = deque(maxlen=nhist)
    delta, dmax = delta0, delta0 * 10
    try:
        while True:
            F = _guard(make_batch(bs, brng), st, budget)
            fx = F(x)
            if mode == 'nomom':
                specials = []
            elif mode == 'rand':
                specials = [d1]
            elif mode == 'gradmom':
                specials = [d1, mom]
            elif mode == 'hist':
                specials = [d1] + list(hist)
            elif mode == 'oracle':
                specials = [d1, g_full(x)]
            else:
                raise ValueError(mode)
            B = _build_basis(n, p, specials, rng)
            fp = np.array([F(x + delta * B[i]) for i in range(p)])
            fm = np.array([F(x - delta * B[i]) for i in range(p)])
            g = (fp - fm) / (2 * delta)
            ghat = B.T @ g
            mom = beta * mom + (1 - beta) * ghat
            hist.append(ghat.copy())
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


def mezo(f_full, g_full, make_batch, x0, f0, budget, bs, lr, eps=1e-2, seed=0):
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


SEEDS = [0, 1, 2]
LRS = [10.0 ** k for k in range(-3, 3)]
DELTAS = [1e-2, 1e-1, 1.0]
BETAS = [0.9, 0.99]


def run_grid(runner, grid, args, f0):
    out = []
    for hp in grid:
        res = [runner(*args, seed=s, **hp) for s in SEEDS]
        loss = float(np.mean([r[0] for r in res])) / f0
        hits = [r[1] for r in res if r[1] is not None]
        hit = int(np.mean(hits)) if len(hits) == len(res) else None
        out.append((loss, hit, hp))
    return min(out, key=lambda t: t[0])


if __name__ == '__main__':
    n = int(os.environ.get('N', 2000))
    budget = int(os.environ.get('BUD', 4000))
    p = int(os.environ.get('P', 20))
    r, m = 10, 4000
    f_full, g_full, mb, x0 = make_problem(n, r, m)
    f0 = f_full(x0)
    print('surrogate: n=%d r=%d m=%d p=%d budget=%d seeds=%d' %
          (n, r, m, p, budget, len(SEEDS)))
    print('metric: final full-sample loss normalized by f(x0); '
          'hit = evaluations to reach 0.5*f0')
    print()
    for bs in [512, 64, 8]:
        print('--- batch=%d ---' % bs)
        args = (f_full, g_full, mb, x0, f0, budget, bs)
        rows = []
        v, hit, hp = run_grid(mezo, [dict(lr=v) for v in LRS], args, f0)
        rows.append(('mezo', v, hit, hp))
        for mode, grid in [
                ('nomom', [dict(delta0=d) for d in DELTAS]),
                ('rand', [dict(delta0=d) for d in DELTAS]),
                ('gradmom', [dict(delta0=d, beta=be)
                             for d in DELTAS for be in BETAS]),
                ('hist', [dict(delta0=d) for d in DELTAS]),
                ('oracle', [dict(delta0=d) for d in DELTAS])]:
            v, hit, hp = run_grid(
                lambda *a, _m=mode, **k: psub(*a, mode=_m, p=p, **k),
                grid, args, f0)
            rows.append((mode, v, hit, hp))
        for name, v, hit, hp in rows:
            print('  %-10s loss=%-11.3e hit=%-7s %s' %
                  (name, v, hit if hit is not None else '-', hp), flush=True)
        d = dict((nm, l) for nm, l, _, _ in rows)
        print('  ratios: nomom/rand=%.2fx gradmom/rand=%.2fx hist/rand=%.2fx '
              'oracle/rand=%.2fx' %
              (d['nomom'] / d['rand'], d['rand'] / d['gradmom'],
               d['rand'] / d['hist'], d['rand'] / d['oracle']))
        print()
