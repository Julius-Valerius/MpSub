"""Equal-budget surrogate study: reuse of function evaluations across steps.

Finite-sum least squares in n dimensions with effective Hessian rank r and
genuine minibatch subsampling. The recycling variant holds one minibatch for
K steps, keeps a sliding window of W evaluated points, buys only q new
directions per step, and fits the subspace gradient by minimum-norm least
squares over the whole window; the window is cleared when the batch changes.
Compared at an equal evaluation budget with per-method tuning against MeZO
and against fresh-direction subspace steps at p in {20, 5}.

Usage: python surrogate_eval_recycle.py
       N=8000 BUD=8000 python surrogate_eval_recycle.py
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


def psub_fresh(f_full, make_batch, x0, f0, budget, bs, delta0,
               p=20, eta=0.1, seed=0):
    """New minibatch and new basis every step; B[0] is momentum; 2p+2 evals/step."""
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


def psub_recycle(f_full, make_batch, x0, f0, budget, bs, delta0,
                 q=2, W=48, K=8, eta=0.1, seed=0,
                 intercept=False, rad=None):
    """Hold one batch for K steps with a sliding point set; 2q+1 evals/step.

    intercept: affine fit df ~ D g + c, the intercept absorbing centre noise.
    rad      : keep only window points with ||y - x|| <= rad * delta.
    """
    n = x0.size
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 8888)
    st, track = _tracker(f_full, f0)
    x = np.array(x0, float)
    track(x)
    delta, dmax = delta0, delta0 * 10
    try:
        while True:
            F = _guard(make_batch(bs, brng), st, budget)
            fx = F(x)                       # one centre value per batch window
            pts_y, pts_f = [], []
            for _ in range(K):
                for _ in range(q):
                    v = rng.standard_normal(n)
                    v /= np.linalg.norm(v)
                    pts_y.append(x + delta * v); pts_f.append(F(pts_y[-1]))
                    pts_y.append(x - delta * v); pts_f.append(F(pts_y[-1]))
                if len(pts_y) > W:
                    pts_y, pts_f = pts_y[-W:], pts_f[-W:]
                D = np.asarray(pts_y) - x
                df = np.asarray(pts_f) - fx
                if rad is not None:
                    keep = np.linalg.norm(D, axis=1) <= rad * delta
                    if keep.sum() >= 2:
                        D, df = D[keep], df[keep]
                # Minimum-norm solution puts g in span(D), so the effective
                # subspace dimension equals the rank of the window.
                if intercept:
                    Dc = np.hstack([D, np.ones((len(D), 1))])
                    sol, *_ = np.linalg.lstsq(Dc, df, rcond=None)
                    g = sol[:-1]
                else:
                    g, *_ = np.linalg.lstsq(D, df, rcond=None)
                gn = np.linalg.norm(g)
                if gn < 1e-300:
                    continue
                s = -delta * g / gn
                xn = x + s
                fn = F(xn)
                pts_y.append(xn); pts_f.append(fn)
                pred = -delta * gn
                rho = (fn - fx) / pred if (pred < -1e-14 and fn < fx) else -1.0
                if fn < fx:
                    x, fx = xn, fn
                    track(x)
                delta = min(delta * 2, dmax) if rho >= eta else max(delta * 0.5, 1e-14)
    except Stop:
        pass
    return st['best'], st['hit']


SEEDS = [0, 1, 2]
LRS = [10.0 ** k for k in range(-3, 3)]
DELTAS = [1e-2, 1e-1, 1.0]


def run_grid(runner, grid, f_full, mb, x0, f0, budget, bs):
    out = []
    for hp in grid:
        res = [runner(f_full, mb, x0, f0, budget, bs, seed=s, **hp) for s in SEEDS]
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
    methods = [
        ('mezo_2', mezo, [dict(lr=v) for v in LRS]),
        ('fresh_p20_42', lambda *a, **k: psub_fresh(*a, p=20, **k),
         [dict(delta0=v) for v in DELTAS]),
        ('fresh_p5_12', lambda *a, **k: psub_fresh(*a, p=5, **k),
         [dict(delta0=v) for v in DELTAS]),
        ('recycle_q2_5', psub_recycle, [dict(delta0=v) for v in DELTAS]),
        ('recycle_icpt_r4',
         lambda *a, **k: psub_recycle(*a, intercept=True, rad=4.0, **k),
         [dict(delta0=v) for v in DELTAS]),
    ]
    for bs in [512, 64, 8]:
        print('--- batch=%d ---' % bs)
        rows = []
        for name, fn, grid in methods:
            loss, hit, hp = run_grid(fn, grid, f_full, mb, x0, f0, budget, bs)
            rows.append((name, loss, hit, hp))
            print('  %-18s loss=%-11.3e hit=%-7s %s' %
                  (name, loss, hit if hit is not None else '-', hp), flush=True)
        base = dict((nm, l) for nm, l, _, _ in rows)
        print('  ratios: recycle/fresh_p20=%.2fx recycle/mezo=%.2fx '
              'recycle/fresh_p5=%.2fx' %
              (base['fresh_p20_42'] / base['recycle_q2_5'],
               base['mezo_2'] / base['recycle_q2_5'],
               base['fresh_p5_12'] / base['recycle_q2_5']))
        print()
