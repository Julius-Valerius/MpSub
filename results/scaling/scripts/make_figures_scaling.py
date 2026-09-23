"""Model-scaling figure: default-radius MpSub against tuned MeZO across sizes.

Reads results/<model-tag>/ directories produced by run_scaling.py and writes
fig_scaling.pdf. Only models that actually have completed logs are plotted, so
the script is useful while the sweep is still running.

Style note. The palette, rcParams and error-bar convention are copied verbatim
from paper/make_figures.py so this figure sits in the same family as the three
already in the paper: teal is always MpSub, gold is always MeZO, and colour
follows the method rather than its rank. The pair was checked for colour-vision
separation (worst adjacent dE 19.9 under protanopia, 25.9 normal, both well
above the dE 8 floor). Gold falls below 3:1 contrast against white, so every
point additionally carries a printed value and a distinct marker shape, and the
same numbers appear in the summary table -- identity and magnitude are never
carried by colour alone.

    python make_figures_scaling.py                 # results/ -> .
    python make_figures_scaling.py results out_dir
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import collect, _rows                       # noqa: E402

TEAL, RED, GOLD = '#0F7B8A', '#A23838', '#C5951D'

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

ORDER = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b']
LABEL = {'opt-125m': 'OPT-125M', 'opt-350m': 'OPT-350M',
         'opt-1.3b': 'OPT-1.3B', 'opt-2.7b': 'OPT-2.7B'}
DEFAULT_RADIUS = '1e-1'


def load_model_dir(d):
    """Return (mpsub default stats, tuned mezo stats, n_lr, budget) or None."""
    mezo = collect(d, 'mezo_lr*_s*.log', 'lr')
    psub = collect(d, 'psub_d*_s*.log', 'd')
    mrows = _rows(mezo, 0)
    prows = dict((r[0], r) for r in _rows(psub, 0))
    if not mrows or DEFAULT_RADIUS not in prows:
        return None
    best = max(mrows, key=lambda r: r[1])
    man_path = os.path.join(d, 'manifest.json')
    budget = 8400
    if os.path.isfile(man_path):
        budget = json.load(open(man_path)).get(
            'budget_forward_passes_requested', 8400)
    return prows[DEFAULT_RADIUS], best, len(mrows), budget


def _err(row):
    """(mean, [[down],[up]]) from a (hp, mean, lo, hi, n) row."""
    _, mean, lo, hi, _ = row
    return mean, np.array([[mean - lo], [hi - mean]])


def fig_scaling(models, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    x = np.arange(len(models))
    tags = [t for t, _ in models]

    # ---- left: accuracy against model size -----------------------------
    ax = axes[0]
    for rows_idx, colour, marker, label, dx, dy in [
            (0, TEAL, 'o', r'MpSub, default $\Delta_0$', -0.11, -15),
            (1, GOLD, 's', r'MeZO, tuned $\eta$', +0.11, +10)]:
        means, errs = [], [[], []]
        for _, payload in models:
            m, e = _err(payload[rows_idx])
            means.append(m)
            errs[0].append(e[0][0])
            errs[1].append(e[1][0])
        # No connecting line: two points joined by a segment read as a
        # trend, and the seed ranges here overlap almost completely.
        ax.errorbar(x + dx, means, yerr=np.array(errs), color=colour,
                    marker=marker, markersize=6, linestyle='none',
                    elinewidth=1.4, capsize=3, label=label)
        # Direct value labels are not decoration here: gold falls below 3:1
        # contrast against white, so magnitude must also be readable as text.
        for xi, m, lo, hi in zip(x + dx, means,
                                 [r[2] for _, p_ in models for r in [p_[rows_idx]]],
                                 [r[3] for _, p_ in models for r in [p_[rows_idx]]]):
            anchor = lo if dy < 0 else hi
            ax.annotate('%.3f' % m, (xi, anchor), textcoords='offset points',
                        xytext=(0, dy), ha='center', fontsize=8, color='0.3')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(t, t) for t in tags])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.margins(y=0.22)
    ax.set_ylabel('test accuracy')
    ax.set_title('Accuracy at matched compute', fontweight='bold')
    ax.text(0.5, -0.24, 'bars: min-max over 3 seeds', transform=ax.transAxes,
            ha='center', fontsize=8, color='0.45')
    ax.legend(loc='best')

    # ---- right: total forward passes to obtain that accuracy -----------
    ax = axes[1]
    w = 0.34
    mp_cost = [payload[3] for _, payload in models]
    mz_cost = [payload[3] * payload[2] for _, payload in models]
    ax.bar(x - w / 2, mp_cost, w, color=TEAL, label='MpSub (1 configuration)')
    ax.bar(x + w / 2, mz_cost, w, color=GOLD,
           label='MeZO (learning-rate grid)')
    for xi, v in zip(x - w / 2, mp_cost):
        ax.annotate('%dk' % round(v / 1000), (xi, v), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=8, color='0.25')
    for xi, v, npts in zip(x + w / 2, mz_cost,
                           [payload[2] for _, payload in models]):
        ax.annotate('%dk' % round(v / 1000), (xi, v),
                    textcoords='offset points', xytext=(0, 3), ha='center',
                    fontsize=8, color='0.25')
        # The 350M grid is smaller than the 125M one; without this label the
        # shorter bar reads as MeZO getting cheaper with scale.
        ax.annotate('%d lr' % npts, (xi, v / 2), ha='center', va='center',
                    fontsize=8, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(t, t) for t in tags])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylabel('forward passes including search')
    ax.set_title('Cost of reaching it', fontweight='bold')
    ax.set_ylim(0, max(mz_cost) * 1.42)
    ax.legend(loc='upper left', ncol=1, handlelength=1.4,
              borderaxespad=0.3, labelspacing=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_scaling.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'results'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    os.makedirs(out_dir, exist_ok=True)
    found = []
    for tag in ORDER:
        d = os.path.join(root, tag)
        if not os.path.isdir(d):
            continue
        payload = load_model_dir(d)
        if payload is None:
            print('[skip] %s has no complete MpSub-default / MeZO pair yet' % tag)
            continue
        found.append((tag, payload))
    if not found:
        sys.exit('no model directory under %s has both a default-radius MpSub '
                 'result and at least one MeZO result yet' % root)
    print('plotting: %s' % ', '.join(t for t, _ in found))
    for tag, payload in found:
        print('  %-9s MpSub %.3f | MeZO best %.3f over %d lr | %d passes/config'
              % (tag, payload[0][1], payload[1][1], payload[2], payload[3]))
    print('wrote %s' % fig_scaling(found, out_dir))


if __name__ == '__main__':
    main()
