"""Summarize a results directory produced by run_scaling.py.

Reads the logs of one model, groups them by hyperparameter, and prints the
mean and min--max range of test and development accuracy over seeds, plus the
comparison the paper makes: default-radius MpSub against the best MeZO
learning rate, and the total forward-pass cost each side paid to get there.

    python summarize.py results/opt-350m
    python summarize.py results/opt-350m results/opt-125m     # several models
"""
import glob
import json
import os
import re
import statistics as st
import sys

ACC = re.compile(r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}")
EVAL_LOSS = re.compile(r"'eval_loss':\s*([0-9.]+)")


def read_log(path):
    """Return (final (test, dev) accuracy or None, eval-loss curve)."""
    acc, curve = None, []
    with open(path, errors='ignore') as fh:
        for line in fh:
            m = ACC.search(line)
            if m:
                acc = (float(m.group(1)), float(m.group(2)))
            m = EVAL_LOSS.search(line)
            if m:
                curve.append(float(m.group(1)))
    return acc, curve


def collect(log_dir, pattern, key):
    groups = {}
    for f in sorted(glob.glob(os.path.join(log_dir, pattern))):
        m = re.search(key + r'([0-9e.+-]+)_s(\d+)\.log$', os.path.basename(f))
        if not m:
            continue
        acc, curve = read_log(f)
        groups.setdefault(m.group(1), []).append((int(m.group(2)), acc, curve))
    return groups


def _rows(groups, index):
    out = []
    for hp, runs in groups.items():
        vals = [a[index] for _, a, _ in runs if a is not None]
        if vals:
            out.append((hp, st.mean(vals), min(vals), max(vals), len(vals)))
    return out


def report(log_dir):
    log_dir = os.path.abspath(log_dir)
    man_path = os.path.join(log_dir, 'manifest.json')
    man = json.load(open(man_path)) if os.path.isfile(man_path) else {}
    print()
    print('=' * 72)
    print('results: %s' % log_dir)
    if man:
        print('model %s | preset %s | budget %d passes/config '
              '| MpSub %d x %d | MeZO %d x 2'
              % (man.get('model'), man.get('preset'),
                 man.get('budget_forward_passes_requested', 0),
                 man.get('psub_steps', 0),
                 man.get('psub_forward_passes_per_step', 0),
                 man.get('mezo_steps', 0)))
    print('=' * 72)

    mezo = collect(log_dir, 'mezo_lr*_s*.log', 'lr')
    psub = collect(log_dir, 'psub_d*_s*.log', 'd')
    best = {}
    for name, groups, label in [('MeZO', mezo, 'lr'), ('MpSub', psub, 'delta0')]:
        rows = _rows(groups, 0)
        if not rows:
            print('%-6s no completed runs' % name)
            print()
            continue
        print('%-6s %-9s %-22s %-22s' % (name, label, 'test acc', 'dev acc'))
        dev = dict((r[0], r) for r in _rows(groups, 1))
        for hp, mean, lo, hi, n in sorted(rows, key=lambda r: -r[1]):
            d = dev.get(hp)
            dtxt = ('%.3f (%.3f-%.3f)' % (d[1], d[2], d[3])) if d else '-'
            print('       %-9s %-22s %-22s n=%d'
                  % (hp, '%.3f (%.3f-%.3f)' % (mean, lo, hi), dtxt, n))
        b = max(rows, key=lambda r: r[1])
        best[name] = b
        print('       best %s=%s at %.3f | grid of %d points'
              % (label, b[0], b[1], len(rows)))
        print()

    budget = man.get('budget_forward_passes_requested', 8400)
    if 'MeZO' in best and '1e-1' in psub:
        d01 = [a[0] for _, a, _ in psub['1e-1'] if a is not None]
        if d01:
            pm, mm = st.mean(d01), best['MeZO'][1]
            n_lr = len(_rows(mezo, 0))
            print('default-radius MpSub  %.3f   (1 configuration, %d passes)'
                  % (pm, budget))
            print('tuned MeZO            %.3f   (%d configurations, %d passes)'
                  % (mm, n_lr, n_lr * budget))
            print('difference            %+.3f' % (pm - mm))
            print('search-cost ratio     %dx' % n_lr)
            print('criterion (>= -0.02)  %s'
                  % ('met' if pm - mm >= -0.02 else 'NOT met'))
            print()


def main():
    dirs = sys.argv[1:]
    if not dirs:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results')
        dirs = sorted(d for d in glob.glob(os.path.join(root, '*'))
                      if os.path.isdir(d))
    if not dirs:
        sys.exit('no results directories found; pass one explicitly')
    for d in dirs:
        report(d)


if __name__ == '__main__':
    main()
