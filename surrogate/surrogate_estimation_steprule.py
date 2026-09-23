"""Equal-budget surrogate study: gradient estimator crossed with step rule.

Finite-sum least squares in n dimensions with effective Hessian rank r and
genuine minibatch subsampling. Each sample loss carries a quartic term,
l_i = 0.5 r_i^2 + qc r_i^4, so that third derivatives do not vanish; on a
purely quadratic objective the debiased estimator below coincides exactly
with central differences and the comparison would be empty.

Estimator layer, all at 2p + 2 evaluations per step:
    central  g_i = (f(x + d b_i) - f(x - d b_i)) / (2d)
    debias   one-sided points at d and 2d, first-order coefficient of the
             quadratic fit, (-3 f0 + 4 f1 - f2) / (2d)
    nodrift  same two points, linear least squares, (f1 - f0 + 2(f2 - f0)) / (5d)

Step-length layer:
    tr   trust region, ||s|| = delta, delta doubled on success and halved
         otherwise
    sig  regularized closed form, ||s|| = sqrt(||g|| / sigma), sigma halved
         on success and doubled otherwise; sampling radius fixed at ds

Usage: python surrogate_estimation_steprule.py
       NONQ=0 python surrogate_estimation_steprule.py
       N=8000 BUD=8000 SDELTA=1.0 python surrogate_estimation_steprule.py
"""
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Stop(Exception):
    pass


def make_problem(n, r, m, seed=0, iso=0.05, qc=0.05):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    A = rng.standard_normal((m, r)) @ U.T + iso * rng.standard_normal((m, n)) / np.sqrt(n)
    wstar = U @ rng.standard_normal(r)
    b = A @ wstar + 0.01 * rng.standard_normal(m)

    def _loss(res):
        return float(np.mean(0.5 * res ** 2 + qc * res ** 4))

    def f_full(w):
        return _loss(A @ np.asarray(w, float) - b)

    def make_batch(bs, brng):
        if bs >= m:
            return f_full
        idx = brng.integers(0, m, bs)
        As, bb = A[idx].copy(), b[idx].copy()

        def fb(w):
            return _loss(As @ np.asarray(w, float) - bb)
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


def _basis(n, p, d1, rng):
    B = np.empty((p, n))
    B[0] = d1 / (np.linalg.norm(d1) + 1e-300)
    for i in range(1, p):
        v = rng.standard_normal(n)
        v -= B[:i].T @ (B[:i] @ v)
        B[i] = v / np.linalg.norm(v)
    return B


def _grad_central(F, x, B, p, ds):
    fp = np.array([F(x + ds * B[i]) for i in range(p)])
    fm = np.array([F(x - ds * B[i]) for i in range(p)])
    return (fp - fm) / (2 * ds)


def _grad_oneside(F, x, fx, B, p, ds, debias):
    f1 = np.array([F(x + ds * B[i]) for i in range(p)])
    f2 = np.array([F(x + 2 * ds * B[i]) for i in range(p)])
    if debias:
        return (-3.0 * fx + 4.0 * f1 - f2) / (2 * ds)
    return ((f1 - fx) + 2.0 * (f2 - fx)) / (5 * ds)


def psub(f_full, make_batch, x0, f0, budget, bs, p,
         estim='central', steprule='tr', delta0=1e-1, sigma0=1.0,
         ds=1e-1, eta=0.1, seed=0):
    n = x0.size
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 8888)
    st, track = _tracker(f_full, f0)
    x = np.array(x0, float)
    track(x)
    d1 = np.zeros(n)
    d1[0] = 1.0
    delta, dmax = delta0, delta0 * 10
    sigma = sigma0
    try:
        while True:
            F = _guard(make_batch(bs, brng), st, budget)
            fx = F(x)
            B = _basis(n, p, d1, rng)
            samp = delta if steprule == 'tr' else ds
            if estim == 'central':
                g = _grad_central(F, x, B, p, samp)
            else:
                g = _grad_oneside(F, x, fx, B, p, samp, debias=(estim == 'debias'))
            gn = np.linalg.norm(g)
            if gn < 1e-300:
                continue
            slen = delta if steprule == 'tr' else np.sqrt(gn / sigma)
            s = -slen * g / gn
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
            if steprule == 'tr':
                delta = min(delta * 2, dmax) if rho >= eta else max(delta * 0.5, 1e-14)
            else:
                sigma = max(sigma / 2, 1e-12) if rho >= eta else min(sigma * 2, 1e12)
    except Stop:
        pass
    return st['best'], st['hit']


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


SEEDS = [0, 1, 2]
LRS = [10.0 ** k for k in range(-3, 3)]
DELTAS = [1e-2, 1e-1, 1.0]
SIGMAS = [1e-3, 1e-1, 1e1, 1e3]


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
    qc = float(os.environ.get('NONQ', 0.05))
    ds = float(os.environ.get('SDELTA', 1e-1))
    r, m = 10, 4000
    f_full, mb, x0 = make_problem(n, r, m, qc=qc)
    f0 = f_full(x0)
    print('surrogate: n=%d r=%d m=%d budget=%d qc=%g ds=%g seeds=%d' %
          (n, r, m, budget, qc, ds, len(SEEDS)))
    print('metric: final full-sample loss normalized by f(x0); '
          'hit = evaluations to reach 0.5*f0')
    print()
    for bs in [64, 8]:
        mz, mzh, mzhp = grid_best(
            lambda seed=0, **hp: mezo(f_full, mb, x0, f0, budget, bs, seed=seed, **hp),
            [dict(lr=v) for v in LRS], f0)
        print('=== batch=%d ===  mezo: loss=%.3e hit=%s %s' % (bs, mz, mzh, mzhp))
        for p in [6, 10, 20]:
            rows = []
            for name, estim, sr, grid in [
                ('tr_central', 'central', 'tr', [dict(delta0=d) for d in DELTAS]),
                ('tr_debias', 'debias', 'tr', [dict(delta0=d) for d in DELTAS]),
                ('sig_central', 'central', 'sig', [dict(sigma0=v) for v in SIGMAS]),
                ('sig_debias', 'debias', 'sig', [dict(sigma0=v) for v in SIGMAS]),
                ('sig_nodrift', 'nodrift', 'sig', [dict(sigma0=v) for v in SIGMAS]),
            ]:
                v, hit, hp = grid_best(
                    lambda seed=0, _e=estim, _s=sr, **hp: psub(
                        f_full, mb, x0, f0, budget, bs, p,
                        estim=_e, steprule=_s, ds=ds, seed=seed, **hp),
                    grid, f0)
                rows.append((name, v, hit, hp))
            print('  --- p=%d ---' % p)
            for name, v, hit, hp in rows:
                print('    %-13s loss=%-11.3e hit=%-7s %s' %
                      (name, v, hit if hit is not None else '-', hp), flush=True)
            d = dict((nm, l) for nm, l, _, _ in rows)
            # All ratios are (numerator loss)/(denominator loss) and losses are
            # "smaller is better", so a value < 1 means the FIRST name is better.
            print('    ratios: tr_central/sig_central=%.2fx '
                  'tr_central/sig_debias=%.2fx sig_debias/sig_central=%.2fx '
                  'sig_nodrift/sig_debias=%.2fx' %
                  (d['tr_central'] / d['sig_central'],
                   d['tr_central'] / d['sig_debias'],
                   d['sig_debias'] / d['sig_central'],
                   d['sig_nodrift'] / d['sig_debias']))
        print()
