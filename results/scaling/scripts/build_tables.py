"""Logs -> convergence_curves.csv, main_results_per_run.csv, main_results_summary.csv.

    python scripts/build_tables.py results data

Every value is parsed from the run logs and the per-model manifest. The MeZO
learning rate in the summary is chosen on development-set accuracy only; test
accuracy plays no part in the selection.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

ORDER = ['opt-125m', 'opt-350m']
METHODS = [('MeZO', 'mezo_lr*_s*.log', 'lr'), ('MpSub', 'psub_d*_s*.log', 'd')]
DEFAULT_RADIUS = '1e-1'

RE_FINAL = re.compile(r"'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)")
RE_EVAL = re.compile(r"'eval_loss':\s*([0-9.]+).*?'epoch':\s*([0-9.]+)")
RE_TRAIN = re.compile(r"'train_runtime':.*?'epoch':\s*([0-9.]+)")


def parse(path):
    final, evals, epoch_end = None, [], None
    for line in open(path, errors='ignore'):
        m = RE_FINAL.search(line)
        if m:
            final = (float(m.group(1)), float(m.group(2)))
        m = RE_EVAL.search(line)
        if m:
            evals.append((float(m.group(2)), float(m.group(1))))
        m = RE_TRAIN.search(line)
        if m:
            epoch_end = float(m.group(1))
    return final, evals, epoch_end


def runs(root):
    out = []
    for tag in ORDER:
        d = os.path.join(root, tag)
        man_path = os.path.join(d, 'manifest.json')
        if not os.path.isdir(d) or not os.path.isfile(man_path):
            continue
        man = json.load(open(man_path))
        for meth, pat, key in METHODS:
            steps = man['mezo_steps'] if meth == 'MeZO' else man['psub_steps']
            per = (man['mezo_forward_passes_per_step'] if meth == 'MeZO'
                   else man['psub_forward_passes_per_step'])
            for f in sorted(glob.glob(os.path.join(d, pat))):
                m = re.search(key + r'([0-9e.+-]+)_s(\d+)\.log$',
                              os.path.basename(f))
                final, evals, epoch_end = parse(f)
                if not m or final is None or not evals or not epoch_end:
                    continue
                curve = []
                for ep, loss in evals:
                    # epoch is the only per-evaluation position the trainer
                    # prints; the run's final epoch maps it back onto steps.
                    step = round(ep / epoch_end * steps)
                    curve.append((step, step * per, loss))
                out.append({'model': tag, 'method': meth, 'hp': m.group(1),
                            'seed': int(m.group(2)),
                            'forward_budget': steps * per,
                            'final_dev_loss': curve[-1][2],
                            'final_dev_accuracy': final[1],
                            'final_test_accuracy': final[0],
                            'curve': curve})
    return out


def write(path, header, body):
    for r in body:
        if len(r) != len(header):
            raise SystemExit('%s: %d columns in header, %d in a row'
                             % (path, len(header), len(r)))
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    print('wrote %s (%d rows)' % (path, len(body)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'results'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data'
    os.makedirs(out, exist_ok=True)
    rs = runs(root)

    write(os.path.join(out, 'convergence_curves.csv'),
          ['model', 'method', 'hyperparameter', 'seed', 'eval_index',
           'step', 'forward_passes', 'dev_loss'],
          [[r['model'], r['method'], r['hp'], r['seed'], i, s, f, '%.6f' % l]
           for r in rs for i, (s, f, l) in enumerate(r['curve'], 1)])

    write(os.path.join(out, 'main_results_per_run.csv'),
          ['model', 'method', 'hyperparameter', 'seed', 'forward_budget',
           'final_dev_loss', 'final_dev_accuracy', 'final_test_accuracy'],
          [[r['model'], r['method'], r['hp'], r['seed'], r['forward_budget'],
            '%.6f' % r['final_dev_loss'], '%.6f' % r['final_dev_accuracy'],
            '%.6f' % r['final_test_accuracy']] for r in
           sorted(rs, key=lambda r: (ORDER.index(r['model']), r['method'],
                                     r['hp'], r['seed']))])

    groups = {}
    for r in rs:
        groups.setdefault((r['model'], r['method'], r['hp']), []).append(r)

    summary = []
    for tag in ORDER:
        for meth, _, _ in METHODS:
            cand = {hp: v for (m, me, hp), v in groups.items()
                    if m == tag and me == meth}
            if not cand:
                continue
            if meth == 'MpSub':
                if DEFAULT_RADIUS not in cand:
                    continue
                hp, rule = DEFAULT_RADIUS, 'fixed default radius (no search)'
            else:
                # Development accuracy only; ties broken on development loss.
                hp = min(cand, key=lambda h: (
                    -st.mean([r['final_dev_accuracy'] for r in cand[h]]),
                    st.mean([r['final_dev_loss'] for r in cand[h]])))
                rule = 'selected on mean development accuracy'
            rr = cand[hp]
            # MpSub's radius grid is a sensitivity ablation, not a search: the
            # reported configuration is the fixed default, so only MeZO is
            # charged for the points it had to try.
            searched = 1 if meth == 'MpSub' else len(cand)
            row = [tag, meth, hp, rule, len(rr), searched, len(cand),
                   rr[0]['forward_budget'], rr[0]['forward_budget'] * searched]
            for field in ['final_dev_loss', 'final_dev_accuracy',
                          'final_test_accuracy']:
                v = [r[field] for r in rr]
                row += ['%.6f' % st.mean(v), '%.6f' % min(v), '%.6f' % max(v)]
            summary.append(row)

    write(os.path.join(out, 'main_results_summary.csv'),
          ['model', 'method', 'hyperparameter', 'selection_rule', 'n_seeds',
           'grid_points_searched', 'grid_points_available',
           'forward_budget_per_run',
           'forward_passes_including_search',
           'dev_loss_mean', 'dev_loss_min', 'dev_loss_max',
           'dev_accuracy_mean', 'dev_accuracy_min', 'dev_accuracy_max',
           'test_accuracy_mean', 'test_accuracy_min', 'test_accuracy_max'],
          summary)


if __name__ == '__main__':
    main()
