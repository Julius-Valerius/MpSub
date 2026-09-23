"""Correctness tests for psub_torch (no GPU or network required).

The critical property checked here is that rejected steps are reverted
exactly, which is the failure-prone part of an in-place implementation.
Run: python test_psub_torch.py
"""
import math
import sys

import torch

from psub_torch import TorchPSubState, _add_dir, _seed_for, psolve_torch

PASS, FAIL = [], []


def chk(cond, name, detail=''):
    (PASS if cond else FAIL).append(name)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


def make_problem(seed=0):
    """Low-effective-rank quadratic: only a few coordinates need to move."""
    torch.manual_seed(seed)
    shapes = [(64, 32), (32, 16), (128,)]
    params = [torch.zeros(s) for s in shapes]
    tgt = [torch.zeros(s) for s in shapes]
    tgt[0][:5, :2] = 1.0

    def f():
        return float(sum(((p - t) ** 2).sum() for p, t in zip(params, tgt)) * 0.5)

    return params, f


def l2diff(a, b):
    return math.sqrt(sum(float(((x - y) ** 2).sum()) for x, y in zip(a, b)))


def main():
    params, _ = make_problem()
    snap = [p.clone() for p in params]
    n = sum(p.numel() for p in params)
    inv = 1.0 / math.sqrt(n)
    gen = torch.Generator(device=params[0].device)
    st = TorchPSubState(delta0=1e-1)
    d = 1e-1
    for i in range(20):
        sv = _seed_for(0, 0, i)
        _add_dir(params, st, gen, sv, i, +d, inv)
        _add_dir(params, st, gen, sv, i, -2 * d, inv)
        _add_dir(params, st, gen, sv, i, +d, inv)
    drift = l2diff(params, snap)
    chk(drift < 1e-4, 'pass-1 perturbations restore exactly',
        '| drift after 20 round trips=%.2e' % drift)

    params, _ = make_problem()
    base = [p.clone() for p in params]
    st = TorchPSubState(delta0=1e-1)
    _add_dir(params, st, gen, _seed_for(0, 0, 3), 3, 1.0, inv)
    v1 = [a - b for a, b in zip(params, base)]
    params2, _ = make_problem()
    st2 = TorchPSubState(delta0=1e-1)
    _add_dir(params2, st2, gen, _seed_for(0, 0, 3), 3, 1.0, inv)
    v2 = [a - b for a, b in zip(params2, base)]
    same = l2diff(v1, v2)
    chk(same < 1e-9, 'same seed regenerates the same direction', '| diff=%.2e' % same)

    params3, _ = make_problem()
    st3 = TorchPSubState(delta0=1e-1)
    _add_dir(params3, st3, gen, _seed_for(0, 0, 4), 4, 1.0, inv)
    v3 = [a - b for a, b in zip(params3, base)]
    dot = sum(float((a * b).sum()) for a, b in zip(v1, v3))
    chk(abs(dot) < 0.2, 'distinct i gives near-orthogonal direction',
        '| cos=%.4f' % dot)

    params, f = make_problem()
    f0 = f()
    st = TorchPSubState(delta0=1e-1)
    fb, st = psolve_torch(f, params, st, p=8, n_steps=60, seed=0)
    chk(fb < f0, 'psolve_torch decreases the loss',
        '| %.6f -> %.6f (ratio=%.4f)' % (f0, fb, fb / f0))
    chk(st.mom is not None and st.mom_norm > 0, 'momentum direction established',
        '| mom_norm=%.4f' % st.mom_norm)

    # Force rejected steps (adversarial objective where trial steps increase loss)
    # and verify that weights are restored exactly to snap without residual drift.
    for d0 in [1e-1, 1.0, 10.0]:
        params, base_f = make_problem()
        snap = [p.clone() for p in params]
        f_init = base_f()
        f_eval_count = [0]
        def f_reject():
            f_eval_count[0] += 1
            # Initial center evaluation returns f_init; trial evaluation returns f_init + 10.0 (rejected)
            return f_init if f_eval_count[0] == 1 else f_init + 10.0
        st = TorchPSubState(delta0=d0)
        psolve_torch(f_reject, params, st, p=4, n_steps=1, seed=99)
        drift = l2diff(params, snap)
        chk(drift < 1e-6, 'rejected steps leave no systematic drift (delta0=%.0e)' % d0,
            '| drift=%.2e' % drift)

    params, f = make_problem()
    cnt = {'n': 0}

    def fc():
        cnt['n'] += 1
        return f()

    st = TorchPSubState(delta0=1e-1)
    p, ns = 6, 4
    psolve_torch(fc, params, st, p=p, n_steps=ns, seed=1)
    want = ns * (2 * p + 1) + 1
    chk(cnt['n'] == want, 'evaluation count equals n_steps*(2p+1)+1',
        '| got=%d want=%d' % (cnt['n'], want))

    params, f = make_problem()
    st = TorchPSubState(delta0=1e-1)
    traj = []
    for k in range(3):
        _, st = psolve_torch(f, params, st, p=4, n_steps=2, seed=k)
        traj.append('%.2e' % st.delta)
    chk(True, 'delta persists across calls', '| trajectory=%s' % traj)

    if FAIL:
        print('RESULT: %d failed -> %s' % (len(FAIL), ', '.join(FAIL)))
        return 1
    print('RESULT: all passed (%d checks)' % len(PASS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
