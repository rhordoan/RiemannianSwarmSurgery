"""
Statistical Testing for CEC 2022 Benchmark Results.

Provides:
- Wilcoxon signed-rank test (pairwise: RSS vs each baseline)
- Friedman test + Nemenyi post-hoc (multi-algorithm ranking)
- LaTeX table generation with statistical significance markers

Usage:
    python benchmarks/statistics.py --results results/
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('Statistics')


def load_results(results_dir, algorithms=None):
    """
    Load per-run results from convergence CSV files.

    Returns:
        dict: {algorithm: {(func, dim): [final_errors]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    for fname in os.listdir(results_dir):
        if not fname.endswith('.csv'):
            continue
        if 'summary' in fname or 'ablation' in fname:
            continue

        # Parse filename: Algorithm_F{num}_D{dim}_run{id}.csv
        parts = fname.replace('.csv', '').split('_')
        if len(parts) < 4:
            continue

        try:
            # Find F and D markers
            algo_parts = []
            func_num = dim = run_id = None
            for p in parts:
                if p.startswith('F') and p[1:].isdigit():
                    func_num = int(p[1:])
                elif p.startswith('D') and p[1:].isdigit():
                    dim = int(p[1:])
                elif p.startswith('run') and p[3:].isdigit():
                    run_id = int(p[3:])
                else:
                    algo_parts.append(p)

            if func_num is None or dim is None or run_id is None:
                continue

            algo = '_'.join(algo_parts) if algo_parts else 'Unknown'

            if algorithms is not None and algo not in algorithms:
                continue

            # Read final error (last row)
            filepath = os.path.join(results_dir, fname)
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    final_error = float(last_row[1])
                    data[algo][(func_num, dim)].append(final_error)
        except (ValueError, IndexError):
            continue

    return dict(data)


def wilcoxon_test(data, reference='RSS-Sheaf', alpha=0.05):
    """
    Pairwise Wilcoxon signed-rank test: reference vs each other algorithm.

    Returns:
        list of dicts with test results.
    """
    from scipy.stats import wilcoxon

    if reference not in data:
        logger.warning(f"Reference algorithm '{reference}' not found in data.")
        return []

    results = []
    ref_data = data[reference]

    for algo, algo_data in data.items():
        if algo == reference:
            continue

        for key in ref_data:
            if key not in algo_data:
                continue

            ref_errors = np.array(ref_data[key])
            algo_errors = np.array(algo_data[key])

            # Ensure same number of runs
            n = min(len(ref_errors), len(algo_errors))
            if n < 5:
                continue

            ref_errors = ref_errors[:n]
            algo_errors = algo_errors[:n]

            # Wilcoxon test
            diff = ref_errors - algo_errors
            if np.all(diff == 0):
                p_value = 1.0
                stat = 0
            else:
                try:
                    stat, p_value = wilcoxon(ref_errors, algo_errors)
                except Exception:
                    p_value = 1.0
                    stat = 0

            # Determine significance marker
            if p_value < alpha:
                if np.mean(ref_errors) < np.mean(algo_errors):
                    marker = '+'  # Reference wins
                else:
                    marker = '-'  # Reference loses
            else:
                marker = '='  # No significant difference

            func_num, dim = key
            results.append({
                'reference': reference,
                'algorithm': algo,
                'function': func_num,
                'dimension': dim,
                'ref_mean': np.mean(ref_errors),
                'algo_mean': np.mean(algo_errors),
                'p_value': p_value,
                'statistic': stat,
                'marker': marker,
            })

    return results


def friedman_test(data):
    """
    Friedman test for multi-algorithm comparison.

    Returns:
        (statistic, p_value, rankings_dict)
    """
    from scipy.stats import friedmanchisquare

    # Collect all (func, dim) keys present in all algorithms
    algos = list(data.keys())
    if len(algos) < 3:
        logger.warning("Friedman test requires at least 3 algorithms.")
        return None, None, {}

    all_keys = set.intersection(*[set(d.keys()) for d in data.values()])

    if not all_keys:
        logger.warning("No common function/dimension pairs found.")
        return None, None, {}

    # Build rank matrix
    rankings = defaultdict(list)

    for key in sorted(all_keys):
        means = {}
        for algo in algos:
            errors = data[algo].get(key, [])
            if errors:
                means[algo] = np.mean(errors)

        if len(means) < len(algos):
            continue

        # Rank algorithms (1 = best)
        sorted_algos = sorted(means.keys(), key=lambda a: means[a])
        for rank, algo in enumerate(sorted_algos, 1):
            rankings[algo].append(rank)

    if not rankings:
        return None, None, {}

    # Average ranks
    avg_ranks = {algo: np.mean(ranks) for algo, ranks in rankings.items()}

    # Friedman test
    n_problems = len(list(rankings.values())[0])
    if n_problems < 2:
        return None, None, avg_ranks

    rank_arrays = [np.array(rankings[algo]) for algo in algos]
    try:
        stat, p_value = friedmanchisquare(*rank_arrays)
    except Exception:
        stat, p_value = None, None

    return stat, p_value, avg_ranks


def generate_latex_table(data, wilcoxon_results, output_path):
    """
    Generate LaTeX table with results and significance markers.

    Format:
        Function | RSS Mean (Std) | Baseline1 Mean (Std) [marker] | ...
    """
    algos = sorted(data.keys())
    all_keys = set()
    for algo_data in data.values():
        all_keys.update(algo_data.keys())
    all_keys = sorted(all_keys)

    # Build marker lookup
    markers = {}
    for wr in wilcoxon_results:
        key = (wr['algorithm'], wr['function'], wr['dimension'])
        markers[key] = wr['marker']

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{CEC 2022 Results: Mean Error (Std)}")
    lines.append(r"\label{tab:results}")

    col_spec = "l" + "c" * len(algos)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = "Function"
    for algo in algos:
        header += f" & {algo}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for func_num, dim in all_keys:
        row = f"F{func_num} (D={dim})"
        for algo in algos:
            errors = data[algo].get((func_num, dim), [])
            if errors:
                mean = np.mean(errors)
                std = np.std(errors)
                marker = markers.get((algo, func_num, dim), '')
                cell = f"{mean:.2e} ({std:.2e})"
                if marker:
                    cell += f" {marker}"
                row += f" & {cell}"
            else:
                row += " & N/A"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    logger.info(f"LaTeX table saved to {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description='Statistical Testing')
    parser.add_argument('--results', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--reference', type=str, default='RSS-Sheaf',
                        help='Reference algorithm for Wilcoxon test')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for tables')
    args = parser.parse_args()

    # Load results
    data = load_results(args.results)

    if not data:
        logger.error(f"No results found in {args.results}")
        return

    logger.info(f"Loaded results for {len(data)} algorithms:")
    for algo, algo_data in data.items():
        total_runs = sum(len(v) for v in algo_data.values())
        logger.info(f"  {algo}: {len(algo_data)} configs, {total_runs} runs")

    # Wilcoxon tests
    logger.info("\n--- Wilcoxon Signed-Rank Tests ---")
    wilcoxon_results = wilcoxon_test(data, reference=args.reference)
    for wr in wilcoxon_results:
        logger.info(
            f"  {wr['reference']} vs {wr['algorithm']} on "
            f"F{wr['function']} D={wr['dimension']}: "
            f"p={wr['p_value']:.4f} [{wr['marker']}] "
            f"(ref={wr['ref_mean']:.4e}, other={wr['algo_mean']:.4e})"
        )

    # Summary: count wins/ties/losses
    if wilcoxon_results:
        opponents = set(wr['algorithm'] for wr in wilcoxon_results)
        for opp in opponents:
            opp_results = [wr for wr in wilcoxon_results
                           if wr['algorithm'] == opp]
            wins = sum(1 for wr in opp_results if wr['marker'] == '+')
            ties = sum(1 for wr in opp_results if wr['marker'] == '=')
            losses = sum(1 for wr in opp_results if wr['marker'] == '-')
            logger.info(
                f"  vs {opp}: {wins}+/{ties}=/{losses}- "
                f"(out of {len(opp_results)})"
            )

    # Friedman test
    logger.info("\n--- Friedman Test ---")
    stat, p_value, avg_ranks = friedman_test(data)
    if stat is not None:
        logger.info(f"  Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        logger.info("  Average rankings:")
        for algo, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
            logger.info(f"    {algo}: {rank:.2f}")
    else:
        logger.info("  Could not compute Friedman test.")

    # Generate LaTeX table
    latex_path = os.path.join(args.output, 'results_table.tex')
    generate_latex_table(data, wilcoxon_results, latex_path)


if __name__ == "__main__":
    main()
