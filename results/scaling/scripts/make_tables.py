"""Emit the CSV tables for the MpSub vs MeZO scaling comparison.

    python scripts/make_tables.py results tables
"""
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import read_log

N_TEST, N_DEV = 56, 50
ORDER = ['opt-125m', 'opt-350m']
DEFAULT_RADIUS = '1e-1'


def load(root):
    rows = []
    for tag in ORDER:
        d = os.path.join(root, tag)
        if not os.path.isdir(d):
            continue
        for meth, pat, key in [('MeZO', 'mezo_lr*_s*.log', 'lr'),
                               ('MpSub', 'psub_d*_s*.log', 'd')]:
            for f in sorted(glob.glob(os.path.join(d, pat))):
                m = re.search(key + r'([0-9e.+-]+)_s(\d+)\.log$',
                              os.path.basename(f))
                acc, _ = read_log(f)
                if not m or acc is None:
                    continue
                rows.append({'model': tag, 'method': meth, 'hp': m.group(1),
                             'seed': int(m.group(2)),
                             'test_acc': acc[0], 'dev_acc': acc[1]})
    return rows


def write(path, header, body):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    print('wrote %s (%d rows)' % (path, len(body)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'results'
    out = sys.argv[2] if len(sys.argv) > 2 else 'tables'
    os.makedirs(out, exist_ok=True)
    rows = load(root)

    for r in rows:
        r['test_correct'] = round(r['test_acc'] * N_TEST)
        r['dev_correct'] = round(r['dev_acc'] * N_DEV)
        assert abs(r['test_correct'] / N_TEST - r['test_acc']) < 1e-9
        assert abs(r['dev_correct'] / N_DEV - r['dev_acc']) < 1e-9

    write(os.path.join(out, 'per_seed.csv'),
          ['model', 'method', 'hyperparameter', 'seed', 'test_acc',
           'test_correct_of_56', 'dev_acc', 'dev_correct_of_50'],
          [[r['model'], r['method'], r['hp'], r['seed'],
            '%.6f' % r['test_acc'], r['test_correct'],
            '%.6f' % r['dev_acc'], r['dev_correct']] for r in rows])

    groups = {}
    for r in rows:
        groups.setdefault((r['model'], r['method'], r['hp']), []).append(r)

    cfg = []
    for (mdl, meth, hp), rs in groups.items():
        t = [r['test_acc'] for r in rs]
        d = [r['dev_acc'] for r in rs]
        sd = st.stdev(t) if len(t) > 1 else 0.0
        cfg.append([mdl, meth, hp, len(rs),
                    '%.4f' % st.mean(t), '%.4f' % min(t), '%.4f' % max(t),
                    '%.4f' % sd, '%.2f' % (sd * N_TEST),
                    '%.4f' % st.mean(d), '%.4f' % min(d), '%.4f' % max(d)])
    cfg.sort(key=lambda r: (ORDER.index(r[0]), r[1], -float(r[4])))
    write(os.path.join(out, 'per_config.csv'),
          ['model', 'method', 'hyperparameter', 'n_seeds', 'test_mean',
           'test_min', 'test_max', 'test_sd', 'test_sd_in_examples',
           'dev_mean', 'dev_min', 'dev_max'], cfg)

    # Pooled within-configuration seed variance, restricted to the two
    # operating-point settings; the divergent settings have several times the
    # spread and are not the error term the comparison needs.
    ss, df = 0.0, 0
    op = []
    for (mdl, meth, hp), rs in groups.items():
        if (meth, hp) not in (('MeZO', '1e-6'), ('MpSub', DEFAULT_RADIUS)):
            continue
        s = st.stdev([r['test_acc'] for r in rs])
        ss += s * s * (len(rs) - 1)
        df += len(rs) - 1
        op.append([mdl, meth, hp, len(rs), '%.4f' % s, '%.2f' % (s * N_TEST)])
    pooled = math.sqrt(ss / df)
    sd_diff = pooled * math.sqrt(2)
    op.sort(key=lambda r: (ORDER.index(r[0]), r[1]))
    op.append(['POOLED', '', '', df + len(op), '%.4f' % pooled,
               '%.2f' % (pooled * N_TEST)])
    write(os.path.join(out, 'seed_variance.csv'),
          ['model', 'method', 'hyperparameter', 'n_seeds', 'test_sd',
           'test_sd_in_examples'], op)

    head, paired = [], []
    for mdl in ORDER:
        mz = {hp: rs for (m, me, hp), rs in groups.items()
              if m == mdl and me == 'MeZO'}
        ps = {hp: rs for (m, me, hp), rs in groups.items()
              if m == mdl and me == 'MpSub'}
        if not mz or DEFAULT_RADIUS not in ps:
            continue
        best_lr = max(mz, key=lambda h: st.mean([r['test_acc'] for r in mz[h]]))
        a = {r['seed']: r['test_acc'] for r in ps[DEFAULT_RADIUS]}
        b = {r['seed']: r['test_acc'] for r in mz[best_lr]}
        seeds = sorted(set(a) & set(b))
        diff = [a[s] - b[s] for s in seeds]
        n = len(diff)
        mean = st.mean(diff)
        s_sample = st.stdev(diff)
        man = os.path.join(root, mdl, 'manifest.json')
        budget = json.load(open(man))['budget_forward_passes_requested']

        head.append([mdl, '%.4f' % st.mean([r['test_acc'] for r in ps[DEFAULT_RADIUS]]),
                     best_lr, '%.4f' % st.mean([r['test_acc'] for r in mz[best_lr]]),
                     '%+.4f' % mean, '%+.1f' % (mean * N_TEST), len(mz),
                     budget, budget * len(mz), '%dx' % len(mz)])

        for label, sd, dfree in [('sample_sd_this_model', s_sample, n - 1),
                                 ('pooled_operating_point_sd', sd_diff, df)]:
            se = sd / math.sqrt(n)
            tc = stats.t.ppf(0.975, dfree)
            tstat = mean / se if se else float('nan')
            p = 2 * stats.t.sf(abs(tstat), dfree) if se else float('nan')
            paired.append([mdl, label, n,
                           ';'.join('%+.4f' % x for x in diff),
                           '%+.4f' % mean, '%.6f' % sd, dfree,
                           '%.3f' % tstat, '%.3f' % p,
                           '%+.4f' % (mean - tc * se),
                           '%+.4f' % (mean + tc * se)])

    write(os.path.join(out, 'headline.csv'),
          ['model', 'mpsub_default_radius_acc', 'mezo_best_lr',
           'mezo_best_acc', 'difference', 'difference_in_examples',
           'mezo_grid_points', 'mpsub_total_passes', 'mezo_total_passes',
           'search_cost_ratio'], head)

    write(os.path.join(out, 'paired_stats.csv'),
          ['model', 'error_term', 'n_seeds', 'per_seed_differences',
           'mean_difference', 'sd', 'df', 't', 'p_two_sided',
           'ci95_lo', 'ci95_hi'], paired)


if __name__ == '__main__':
    main()
