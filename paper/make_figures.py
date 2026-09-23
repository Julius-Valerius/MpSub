"""Regenerate the three figures of the paper from the experiment logs.

Reads the logs produced by experiments/run_hp_cost.sh (or its parallel
driver) and writes fig_sensitivity.pdf, fig_convergence.pdf and
fig_p_ablation.pdf next to this file. Every configuration in the logs used
8400 forward passes; MeZO evaluates twice per step and MpSub 2p+2 = 42 times
per step at p = 20, which sets the horizontal axis of the convergence plot.
The p ablation is transcribed from a separate 100-step single-seed sweep.

Usage: python make_figures.py [log_dir] [out_dir]
"""
import os
import re
import sys
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRUSSIAN_BLUE, GOLD = '#003153', '#C5951D'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

MEZO_LRS = ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4']
PSUB_DELTAS = ['1e-3', '1e-2', '3e-2', '1e-1', '3e-1']
MEZO_FWD_PER_STEP = 2
PSUB_FWD_PER_STEP = 42          # 2p + 2 at p = 20
MEZO_EVAL_EVERY = 300           # steps
PSUB_EVAL_EVERY = 25

# p, dev accuracy, test accuracy, seconds per step (RTX 4060 Ti).
P_ABLATION = [(5, 0.70, 0.643, 3.0), (10, 0.72, 0.661, 4.4),
              (15, 0.74, 0.661, 6.0), (20, 0.78, 0.679, 7.7),
              (25, 0.78, 0.679, 8.7), (30, 0.76, 0.661, 10.1)]


def read_log(path):
    """Return (eval_loss curve, final test accuracy, final dev accuracy)."""
    ev, acc = [], None
    with open(path, errors='ignore') as fh:
        for ln in fh:
            m = re.search(r"'eval_loss':\s*([0-9.]+)", ln)
            if m:
                ev.append(float(m.group(1)))
            m = re.search(r"\{'accuracy':\s*([0-9.]+),\s*'dev_accuracy':\s*([0-9.]+)\}", ln)
            if m:
                acc = (float(m.group(1)), float(m.group(2)))
    return ev, acc


def collect(log_dir, pattern):
    """Group logs by hyperparameter value; return {hp: [(curve, acc), ...]}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(log_dir, pattern))):
        m = re.search(r'_(?:lr|d)([0-9e.+-]+)_s(\d+)\.log$', os.path.basename(f))
        if not m:
            continue
        out.setdefault(m.group(1), []).append(read_log(f))
    return out


def _stats(groups, keys, index):
    mean, lo, hi = [], [], []
    for k in keys:
        v = [a[index] for _, a in groups.get(k, []) if a is not None]
        if not v:
            v = [np.nan]
        mean.append(np.mean(v))
        lo.append(np.min(v))
        hi.append(np.max(v))
    mean, lo, hi = np.array(mean), np.array(lo), np.array(hi)
    return mean, np.vstack([mean - lo, hi - mean])


def fig_sensitivity(mezo, psub, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    mz, mzerr = _stats(mezo, MEZO_LRS, 0)
    ps, pserr = _stats(psub, PSUB_DELTAS, 0)
    tuned = np.nanmax(mz)

    ax = axes[0]
    x = np.arange(len(MEZO_LRS))
    ax.errorbar(x, mz, yerr=mzerr, color=GOLD, marker='s', markersize=5,
                linewidth=1.4, capsize=3, label='MeZO')
    ax.axhline(tuned, color='0.35', linestyle=':', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$',
                        r'$10^{-5}$', r'$10^{-4}$'])
    ax.set_xlabel(r'learning rate $\eta$')
    ax.set_ylabel('test accuracy')
    ax.set_title('MeZO', fontweight='bold')

    ax = axes[1]
    x = np.arange(len(PSUB_DELTAS))
    ax.errorbar(x, ps, yerr=pserr, color=PRUSSIAN_BLUE, marker='o', markersize=5,
                linewidth=1.4, capsize=3, label='MpSub')
    ax.axhline(tuned, color='0.35', linestyle=':', linewidth=1.0,
               label='tuned MeZO mean')
    ax.set_xticks(x)
    ax.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$3\times10^{-2}$',
                        r'$10^{-1}$', r'$3\times10^{-1}$'])
    ax.set_xlabel(r'initial radius $\Delta_0$')
    ax.set_title('MpSub', fontweight='bold')
    ax.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sensitivity.pdf'))
    plt.close(fig)


def _curve(runs, eval_every, fwd_per_step):
    """Mean, min and max eval-loss curves on a forward-pass axis."""
    curves = [c for c, _ in runs if c]
    if not curves:
        return None
    k = min(len(c) for c in curves)
    arr = np.array([c[:k] for c in curves])
    x = (np.arange(1, k + 1) * eval_every) * fwd_per_step
    return x, arr.mean(0), arr.min(0), arr.max(0)


def fig_convergence(mezo, psub, out_dir):
    best_lr = max((k for k in mezo),
                  key=lambda k: np.mean([a[0] for _, a in mezo[k] if a]))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for runs, ev, fw, color, marker, label in [
            (mezo[best_lr], MEZO_EVAL_EVERY, MEZO_FWD_PER_STEP, GOLD, 's',
             r'MeZO, tuned $\eta$'),
            (psub['1e-1'], PSUB_EVAL_EVERY, PSUB_FWD_PER_STEP, PRUSSIAN_BLUE, 'o',
             r'MpSub, default $\Delta_0$')]:
        got = _curve(runs, ev, fw)
        if got is None:
            continue
        x, mean, lo, hi = got
        ax.errorbar(x, mean, yerr=np.vstack([mean - lo, hi - mean]),
                    color=color, marker=marker, markersize=4, linewidth=1.4,
                    capsize=3, label=label)
    ax.set_xlabel('forward passes')
    ax.set_ylabel('development loss')
    ax.set_title('Matched-compute convergence\nOPT-125M, CB, full parameters',
                 fontweight='bold')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_convergence.pdf'))
    plt.close(fig)


def fig_p_ablation(out_dir):
    p = [r[0] for r in P_ABLATION]
    dev = [r[1] for r in P_ABLATION]
    test = [r[2] for r in P_ABLATION]
    sec = [r[3] for r in P_ABLATION]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(p, dev, color=PRUSSIAN_BLUE, marker='o', markersize=5, linewidth=1.4,
            label='dev accuracy')
    ax.plot(p, test, color=GOLD, marker='^', markersize=5, linewidth=1.4,
            linestyle='--', label='test accuracy')
    ax.set_xlabel(r'subspace dimension $p$')
    ax.set_ylabel('accuracy')
    ax.set_xticks(p)
    ax.set_title('Subspace dimension\n100 steps, $\\Delta_0=10^{-1}$',
                 fontweight='bold')
    ax.legend(loc='center right')

    ax2 = ax.twinx()
    ax2.plot(p, sec, color=GOLD, marker='s', markersize=4, linewidth=1.2,
             linestyle=':')
    ax2.set_ylabel('seconds per step', color=GOLD)
    ax2.tick_params(axis='y', labelcolor=GOLD)
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_p_ablation.pdf'))
    plt.close(fig)


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'hp_cost')
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(
        os.path.abspath(__file__))
    mezo = collect(log_dir, 'mezo_lr*_s*.log')
    psub = collect(log_dir, 'psub_d*_s*.log')
    if not mezo or not psub:
        raise SystemExit('no logs found under %s' % log_dir)
    fig_sensitivity(mezo, psub, out_dir)
    fig_convergence(mezo, psub, out_dir)
    fig_p_ablation(out_dir)
    print('wrote fig_sensitivity.pdf, fig_convergence.pdf, fig_p_ablation.pdf '
          'to %s' % out_dir)


if __name__ == '__main__':
    main()
