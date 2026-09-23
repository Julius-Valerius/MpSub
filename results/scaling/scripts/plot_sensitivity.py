"""2x2 hyperparameter-sensitivity figure, if both models have complete grids.

    python scripts/plot_sensitivity.py data .

Rows are model sizes, columns are the MeZO learning-rate grid and the MpSub
initial-radius grid. Points are the mean final test accuracy over seeds with
min-max bars. The fixed default radius is ringed; the learning rate selected on
development accuracy is starred. The figure is written only when every model
carries the full grid on both axes -- otherwise the script reports exactly which
runs are missing and writes nothing.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TEAL, RED, GOLD = '#0F7B8A', '#A23838', '#C5951D'
LABEL = {'opt-125m': 'OPT-125M', 'opt-350m': 'OPT-350M'}
ORDER = ['opt-125m', 'opt-350m']
DEFAULT_RADIUS = '1e-1'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.linewidth': 0.8, 'axes.grid': True, 'grid.linestyle': '--',
    'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


def read(data_dir):
    runs = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(data_dir,
                                              'main_results_per_run.csv'))):
        runs[(r['model'], r['method'], r['hyperparameter'])].append(
            (float(r['final_test_accuracy']), float(r['final_dev_accuracy'])))
    chosen = {}
    for r in csv.DictReader(open(os.path.join(data_dir,
                                              'main_results_summary.csv'))):
        chosen[(r['model'], r['method'])] = r['hyperparameter']
    return runs, chosen


def grids(runs):
    g = collections.defaultdict(set)
    for (model, method, hp) in runs:
        g[(model, method)].add(hp)
    return g


def completeness(g):
    """Union of every model's grid per method is what each model must carry."""
    want = collections.defaultdict(set)
    for (model, method), hps in g.items():
        want[method] |= hps
    missing = {}
    for model in ORDER:
        for method in ['MeZO', 'MpSub']:
            gap = want[method] - g.get((model, method), set())
            if gap:
                missing[(model, method)] = sorted(gap, key=float)
    return want, missing


def panel(ax, runs, model, method, hp_list, chosen, colour, marker, xlabel):
    xs = sorted(hp_list, key=float)
    x = np.array([float(h) for h in xs])
    mean = np.array([np.mean([t for t, _ in runs[(model, method, h)]])
                     for h in xs])
    lo = np.array([np.min([t for t, _ in runs[(model, method, h)]]) for h in xs])
    hi = np.array([np.max([t for t, _ in runs[(model, method, h)]]) for h in xs])
    ax.errorbar(x, mean, yerr=np.vstack([mean - lo, hi - mean]), color=colour,
                marker=marker, markersize=5, linewidth=1.4, capsize=3,
                zorder=3)
    ax.set_xscale('log')
    # Pad in log space so the ringed / starred marker at a grid end is not
    # clipped by the spine.
    ax.set_xlim(x.min() / 2.6, x.max() * 2.6)
    ax.set_xlabel(xlabel)
    mark = chosen[(model, method)]
    xm, ym = float(mark), mean[xs.index(mark)]
    if method == 'MpSub':
        ax.plot([xm], [ym], 'o', markersize=12, markerfacecolor='none',
                markeredgecolor=RED, markeredgewidth=1.4, zorder=4,
                label='fixed default radius')
    else:
        ax.plot([xm], [ym], '*', markersize=15, markerfacecolor='none',
                markeredgecolor=RED, markeredgewidth=1.2, zorder=4,
                label='selected on development accuracy')
    ax.legend(loc='lower center', handletextpad=0.2)
    return mean, lo, hi


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    runs, chosen = read(data_dir)
    g = grids(runs)
    want, missing = completeness(g)

    print('grids present')
    for model in ORDER:
        for method in ['MeZO', 'MpSub']:
            have = sorted(g.get((model, method), set()), key=float)
            print('  %-9s %-6s %d/%d  %s'
                  % (model, method, len(have), len(want[method]),
                     ' '.join(have)))
    if missing:
        print()
        print('fig_sensitivity_models.pdf NOT written -- incomplete grids:')
        for (model, method), gap in sorted(missing.items()):
            print('  %s %s missing %s  (%d runs at 3 seeds)'
                  % (model, method, ' '.join(gap), 3 * len(gap)))
        return 1

    fig, axes = plt.subplots(len(ORDER), 2, sharey='row',
                             figsize=(7.4, 2.8 * len(ORDER)),
                             squeeze=False)
    for i, model in enumerate(ORDER):
        panel(axes[i][0], runs, model, 'MeZO', want['MeZO'], chosen, GOLD, 's',
              r'MeZO learning rate $\eta$')
        panel(axes[i][1], runs, model, 'MpSub', want['MpSub'], chosen, TEAL,
              'o', r'MpSub initial radius $\Delta_0$')
        axes[i][0].set_ylabel('%s\nfinal test accuracy' % LABEL[model])
    axes[0][0].set_title('MeZO', fontweight='bold')
    axes[0][1].set_title('MpSub', fontweight='bold')
    fig.text(0.5, -0.03, 'points: mean over 3 seeds; bars: min-max',
             ha='center', fontsize=8, color='0.45')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_sensitivity_models.pdf')
    fig.savefig(path)
    plt.close(fig)
    print('wrote %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
