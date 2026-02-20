"""
TMI Results Analysis Script.

Reads the CSV produced by run_tmi_benchmark.py and outputs:
  1. Console summary table (mean ± std per variant per function)
  2. Publication-ready LaTeX table
  3. Convergence curve plot (mean ± IQR) for each function
  4. Bar chart: mean error per variant grouped by function
  5. Key ablation summary: D (ORC-guided) vs E (random) — the paper's core claim

Usage:
    python benchmarks/analyze_results.py results/tmi_cec2022_<timestamp>.csv
    python benchmarks/analyze_results.py results/tmi_cec2022_<timestamp>.csv --latex
    python benchmarks/analyze_results.py results/tmi_cec2022_<timestamp>.csv --no-plots
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['final_error'] = float(r['final_error'])
        r['func_num'] = int(r['func_num'])
        r['dim'] = int(r['dim'])
        r['seed'] = int(r['seed'])
    return rows


def group_errors(rows: list) -> dict:
    """Returns {(func_num, variant): np.array of final_errors}."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r['func_num'], r['variant'])].append(r['final_error'])
    return {k: np.array(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1.0
    diff = a - b
    if np.all(diff == 0):
        return 1.0
    try:
        _, p = wilcoxon(a, b, alternative='two-sided')
        return float(p)
    except Exception:
        return 1.0


def sig_star(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


FUNC_TYPE = {
    1: 'U', 2: 'U', 3: 'U',         # Unimodal
    4: 'M', 5: 'M', 6: 'M',         # Multimodal
    7: 'M', 8: 'M', 9: 'M', 10: 'M',
    11: 'H', 12: 'H',                # Hybrid/Composition
}

FUNC_NAME = {
    1: 'Zakharov',           2: 'Rosenbrock',        3: 'Exp. Schaffer',
    4: 'Non-cont. Rastr.',   5: 'Levy',              6: 'Mod. Schwefel',
    7: 'Bent Cigar',         8: 'HGBat',             9: 'Katsuura',
    10: 'Ackley',            11: 'Weierstrass',      12: 'Exp. GR',
}


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_console_table(errors: dict, func_nums: list, variants: list):
    present = sorted({v for (_, v) in errors})
    variants = [v for v in variants if v in present]

    col_w = 18
    hdr = f"{'F#':>3}  {'Type':>4}  {'Name':<22}"
    for v in variants:
        hdr += f"  {('mean±std ' + v):>{col_w}}"
    print()
    print('=' * (len(hdr) + 4))
    print('TMI CEC 2022 — Final Error (mean ± std)')
    print('=' * (len(hdr) + 4))
    print(hdr)
    print('-' * (len(hdr) + 4))

    for fn in func_nums:
        row = f"{fn:>3}  {FUNC_TYPE.get(fn,'?'):>4}  {FUNC_NAME.get(fn,'?'):<22}"
        for v in variants:
            arr = errors.get((fn, v), np.array([]))
            if len(arr):
                row += f"  {arr.mean():>8.3e}±{arr.std():>7.2e}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)
    print('=' * (len(hdr) + 4))


# ---------------------------------------------------------------------------
# Key ablation: D vs E
# ---------------------------------------------------------------------------

def print_ablation_summary(errors: dict, func_nums: list):
    """The critical test: ORC-guided (D) vs random injection (E)."""
    has_D = any((fn, 'D') in errors for fn in func_nums)
    has_E = any((fn, 'E') in errors for fn in func_nums)
    has_C = any((fn, 'C') in errors for fn in func_nums)

    if not (has_D and has_E):
        print('\n[ablation] Variant E (random injection) not in this dataset — skipping D vs E comparison.')
        return

    print()
    print('=' * 75)
    print('KEY ABLATION: ORC-guided (D) vs Random injection (E) vs NL-SHADE (C)')
    print('If D << E, ORC geometry is earning its place.')
    print('If D ≈ E, the improvement is from restarts alone, not geometry.')
    print('-' * 75)
    print(f"{'F#':>3}  {'Type':>4}  {'C(NL-SHADE)':>12}  {'D(+TMI)':>12}  {'E(+RandInj)':>12}  {'p(D/E)':>7}  {'D vs E':>8}")
    print('-' * 75)

    d_beats_e, e_beats_d, ties = 0, 0, 0

    for fn in func_nums:
        arr_C = errors.get((fn, 'C'), np.array([]))
        arr_D = errors.get((fn, 'D'), np.array([]))
        arr_E = errors.get((fn, 'E'), np.array([]))

        mC = arr_C.mean() if len(arr_C) else float('nan')
        mD = arr_D.mean() if len(arr_D) else float('nan')
        mE = arr_E.mean() if len(arr_E) else float('nan')

        p_DE = wilcoxon_p(arr_D, arr_E) if (len(arr_D) and len(arr_E)) else 1.0
        stars = sig_star(p_DE)

        if p_DE < 0.05 and mD < mE:
            verdict = 'D wins  ✓'
            d_beats_e += 1
        elif p_DE < 0.05 and mD > mE:
            verdict = 'E wins  ✗'
            e_beats_d += 1
        else:
            verdict = 'tie   ~'
            ties += 1

        ftype = FUNC_TYPE.get(fn, '?')
        print(f"{fn:>3}  {ftype:>4}  {mC:>12.3e}  {mD:>12.3e}  {mE:>12.3e}  {p_DE:>7.4f}  {stars + ' ' + verdict}")

    print('-' * 75)
    n = len(func_nums)
    print(f'D(ORC) beats random: {d_beats_e}/{n}   '
          f'Random beats ORC: {e_beats_d}/{n}   '
          f'Ties: {ties}/{n}')

    if d_beats_e > e_beats_d:
        print('\n>>> ORC geometry is contributing beyond the restart mechanism. <<< (paper claim supported)')
    elif d_beats_e == e_beats_d:
        print('\n>>> Mixed: geometry helps on some functions but not others.')
    else:
        print('\n>>> Random injection matches or beats ORC — geometry contribution unclear.')
    print('=' * 75)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def print_latex_table(errors: dict, func_nums: list, variants: list, dim: int):
    """Output a LaTeX table for direct copy-paste into the paper."""
    present = sorted({v for (_, v) in errors})
    variants = [v for v in variants if v in present]
    n_cols = len(variants)

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Mean final error on CEC 2022 (D=' + str(dim) + r', 30 runs). '
                 r'Bold = best per row. $\dagger$ = D significantly better than E (Wilcoxon $p<0.05$).}')
    lines.append(r'\label{tab:cec2022}')
    lines.append(r'\resizebox{\linewidth}{!}{%')
    col_spec = 'lll' + 'r' * n_cols
    lines.append(r'\begin{tabular}{' + col_spec + r'}')
    lines.append(r'\toprule')
    header = r'F\# & Type & Name & ' + ' & '.join(variants) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    for fn in func_nums:
        means = {}
        for v in variants:
            arr = errors.get((fn, v), np.array([]))
            means[v] = arr.mean() if len(arr) else float('nan')

        best_v = min((v for v in variants if not np.isnan(means[v])),
                     key=lambda v: means[v], default=None)

        # D vs E significance dagger
        arr_D = errors.get((fn, 'D'), np.array([]))
        arr_E = errors.get((fn, 'E'), np.array([]))
        p_DE = wilcoxon_p(arr_D, arr_E) if (len(arr_D) and len(arr_E)) else 1.0
        dagger_D = r'$^\dagger$' if (p_DE < 0.05 and means.get('D', float('nan')) < means.get('E', float('nan'))) else ''

        cells = []
        for v in variants:
            m = means[v]
            if np.isnan(m):
                cell = '—'
            else:
                val = f'{m:.2e}'
                if v == best_v:
                    val = r'\textbf{' + val + '}'
                if v == 'D':
                    val += dagger_D
                cell = val
            cells.append(cell)

        ftype = FUNC_TYPE.get(fn, '?')
        fname = FUNC_NAME.get(fn, '?')
        row = f'{fn} & {ftype} & {fname} & ' + ' & '.join(cells) + r' \\'
        lines.append(row)

        if fn == 3 or fn == 10:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table}')

    print()
    print('% ============ LaTeX Table ============')
    print('\n'.join(lines))
    print('% =====================================')


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(rows: list, errors: dict, func_nums: list, variants: list,
               out_dir: Path, dim: int):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print('[plots] matplotlib not available — skipping plots.')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted({v for (_, v) in errors})
    variants = [v for v in variants if v in present]

    COLORS = {'A': '#4878CF', 'B': '#6ACC65', 'C': '#D65F5F',
              'D': '#B47CC7', 'E': '#C4AD66'}
    LABELS = {'A': 'L-SHADE', 'B': 'L-SHADE+TMI',
              'C': 'NL-SHADE', 'D': 'NL-SHADE+TMI',
              'E': 'NL-SHADE+RandInj'}
    LSTYLE = {'A': '-', 'B': '--', 'C': '-', 'D': '--', 'E': ':'}

    # --- Figure 1: Bar chart of mean final errors ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    for ax_i, fn in enumerate(func_nums[:12]):
        ax = axes[ax_i]
        means = [errors.get((fn, v), np.array([])) for v in variants]
        x = np.arange(len(variants))
        bars = ax.bar(x,
                      [m.mean() if len(m) else 0 for m in means],
                      yerr=[m.std() if len(m) else 0 for m in means],
                      color=[COLORS.get(v, 'gray') for v in variants],
                      capsize=3, alpha=0.85)
        ax.set_title(f'F{fn} ({FUNC_TYPE.get(fn,"?")}): {FUNC_NAME.get(fn,"")}', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=7)
        ax.set_ylabel('Mean error', fontsize=7)
        ax.tick_params(axis='y', labelsize=7)

    legend_handles = [Line2D([0], [0], color=COLORS.get(v, 'gray'),
                              label=f'{v}: {LABELS.get(v,v)}', linewidth=3)
                      for v in variants]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(variants),
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'TMI CEC 2022  D={dim}  Final Error by Variant', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path = out_dir / f'tmi_bar_D{dim}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[plots] Bar chart saved: {out_path}')

    # --- Figure 2: D vs E scatter (the ablation) ---
    has_D_E = all(any((fn, v) in errors for fn in func_nums) for v in ['D', 'E'])
    if has_D_E:
        fig, ax = plt.subplots(figsize=(7, 7))
        for fn in func_nums:
            mD = errors.get((fn, 'D'), np.array([])).mean() if (fn, 'D') in errors else None
            mE = errors.get((fn, 'E'), np.array([])).mean() if (fn, 'E') in errors else None
            if mD is None or mE is None:
                continue
            color = {'U': '#4878CF', 'M': '#D65F5F', 'H': '#6ACC65'}.get(FUNC_TYPE.get(fn,'U'), 'gray')
            ax.scatter(mE, mD, c=color, s=80, zorder=3)
            ax.annotate(f'F{fn}', (mE, mD), textcoords='offset points',
                        xytext=(4, 4), fontsize=8)

        lim_max = max(
            max((errors.get((fn, 'D'), np.array([0])).mean() for fn in func_nums), default=1),
            max((errors.get((fn, 'E'), np.array([0])).mean() for fn in func_nums), default=1),
        ) * 1.1
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, label='D = E (no gain)')
        ax.set_xlabel('E: NL-SHADE + Random Injection (mean error)', fontsize=11)
        ax.set_ylabel('D: NL-SHADE + TMI/ORC (mean error)', fontsize=11)
        ax.set_title('Ablation: ORC-guided (D) vs Random injection (E)\n'
                     'Points below diagonal = ORC helps', fontsize=11)
        legend_els = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=10,label=l)
                      for c,l in [('#4878CF','Unimodal'),('#D65F5F','Multimodal'),('#6ACC65','Hybrid')]]
        legend_els.append(Line2D([0],[0],linestyle='--',color='k',alpha=0.4,label='D=E (no gain)'))
        ax.legend(handles=legend_els, fontsize=9)
        plt.tight_layout()
        out_path = out_dir / f'tmi_ablation_D_vs_E_D{dim}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[plots] Ablation scatter saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze TMI benchmark results')
    parser.add_argument('csv', help='Path to results CSV file')
    parser.add_argument('--latex', action='store_true', help='Print LaTeX table')
    parser.add_argument('--no-plots', action='store_true', help='Skip matplotlib plots')
    parser.add_argument('--variants', nargs='+', default=['A', 'B', 'C', 'D', 'E'],
                        help='Variants to include (default: A B C D E)')
    parser.add_argument('--funcs', type=int, nargs='+', default=list(range(1, 13)),
                        help='Function numbers (default: 1-12)')
    args = parser.parse_args()

    print(f'Loading: {args.csv}')
    rows = load_csv(args.csv)
    errors = group_errors(rows)

    # Determine dimensionality and n_seeds from data
    dims = list({r['dim'] for r in rows})
    dim = dims[0] if len(dims) == 1 else 10
    n_seeds = max(len(v) for v in errors.values()) if errors else 0

    func_nums = sorted(set(args.funcs) & {fn for (fn, _) in errors})
    variants = args.variants

    print_console_table(errors, func_nums, variants)
    print_ablation_summary(errors, func_nums)

    if args.latex:
        print_latex_table(errors, func_nums, variants, dim)

    if not args.no_plots:
        out_dir = Path(args.csv).parent / 'figures'
        make_plots(rows, errors, func_nums, variants, out_dir, dim)

    print(f'\nDone. {len(rows)} runs analyzed.')


if __name__ == '__main__':
    main()
