"""Two-panel development-loss convergence figure.

    python scripts/plot_convergence.py data .

Reads data/convergence_curves.csv and data/main_results_summary.csv only; the
MeZO learning rate drawn is whichever one the summary selected on development
accuracy. Evaluation points are plotted where they were recorded and joined by
straight segments -- no smoothing, no resampling, no iteration axis.

Palette is the paper's existing pair (teal MpSub, gold MeZO). Its adjacent-pair
separation is ok (dE 19.9 protan, 25.9 normal); gold sits below 3:1 against
white and teal is low-chroma, so identity never rests on colour: the two series
also differ in line style, marker shape and band texture, which is what carries
the figure in black and white.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRUSSIAN_BLUE, GOLD = '#003153', '#C5951D'
LABEL = {'opt-125m': 'OPT-125M', 'opt-350m': 'OPT-350M'}
ORDER = ['opt-125m', 'opt-350m']

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.linewidth': 0.8, 'axes.grid': True, 'grid.linestyle': '--',
    'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'hatch.linewidth': 0.3,
})


def mathhp(s):
    """'3e-1' -> '3\\times10^{-1}'; '10.0' -> '10'. Formatting only."""
    if 'e' in s:
        mant, exp = s.split('e')
        mant = mant.rstrip('.0') or '1'
        head = '' if mant == '1' else mant + r'\times '
        return '%s10^{%d}' % (head, int(exp))
    f = float(s)
    return '%g' % f


def read(data_dir):
    curves = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(os.path.join(data_dir,
                                              'convergence_curves.csv'))):
        curves[(r['model'], r['method'], r['hyperparameter'])][
            int(r['seed'])].append((int(r['forward_passes']),
                                    float(r['dev_loss'])))
    chosen = {}
    for r in csv.DictReader(open(os.path.join(data_dir,
                                              'main_results_summary.csv'))):
        chosen[(r['model'], r['method'])] = (r['hyperparameter'],
                                             r['selection_rule'])
    return curves, chosen


def series(curves, key):
    """(x, mean, lo, hi) over seeds; requires identical evaluation grids."""
    per_seed = curves[key]
    grids = {tuple(x for x, _ in sorted(v)) for v in per_seed.values()}
    if len(grids) != 1:
        raise SystemExit('evaluation grids differ across seeds for %s' % (key,))
    x = np.array(sorted(grids)[0], dtype=float)
    y = np.array([[l for _, l in sorted(v)] for v in per_seed.values()])
    mean = y.mean(axis=0)
    sem = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
    lo = mean - sem
    hi = mean + sem
    return x, mean, lo, hi, y.shape[0]


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    curves, chosen = read(data_dir)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    nseed = set()
    for ax, model in zip(axes, ORDER):
        for method, colour, marker, ls, hatch in [
                ('MpSub', PRUSSIAN_BLUE, 'o', '-', None),
                ('MeZO', GOLD, 's', '--', '/////')]:
            hp, rule = chosen[(model, method)]
            x, mean, lo, hi, n = series(curves, (model, method, hp))
            nseed.add(n)
            ax.fill_between(x, lo, hi, facecolor=colour, alpha=0.16,
                            edgecolor='none', linewidth=0.0, zorder=1)
            if hatch:
                # Texture, not colour, is what separates the two bands on a
                # black-and-white print.
                ax.fill_between(x, lo, hi, facecolor='none', edgecolor=colour,
                                linewidth=0.0, hatch=hatch, alpha=0.30,
                                zorder=2)
            tag = (r'MpSub, fixed default $\Delta_0=%s$' % mathhp(hp)
                   if method == 'MpSub'
                   else r'MeZO, dev-selected $\eta=%s$' % mathhp(hp))
            ax.plot(x, mean, color=colour, marker=marker, markersize=4.2,
                    linewidth=1.4, linestyle=ls, markeredgecolor='white',
                    markeredgewidth=0.4, label=tag, zorder=3)
        ax.set_xlim(0, max(x) * 1.02)
        ax.set_xlabel('Training-objective forward passes')
        ax.set_title(LABEL[model], fontweight='bold')
        ax.legend(loc='upper right')
    axes[0].set_ylabel('Development loss')
    if len(nseed) != 1:
        raise SystemExit('seed counts differ between panels: %s' % nseed)
    nseed.pop()
    fig.tight_layout(pad=0.35)
    path = os.path.join(out_dir, 'fig_convergence_models.pdf')
    fig.savefig(path)
    fig.savefig(os.path.join(out_dir, 'fig_convergence_models.png'), dpi=300)
    plt.close(fig)
    print('wrote %s' % path)


if __name__ == '__main__':
    main()
