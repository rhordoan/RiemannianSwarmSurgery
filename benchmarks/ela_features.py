"""
Compute discrete ELA-style features for NK and W-model landscapes and
correlate with algorithm performance.

Features computed:
  - Nearest-better clustering (NBC): ratio of mean distance to nearest-better
    solution vs. mean distance to nearest neighbor (among sampled solutions).
  - Fitness cloud skewness: skewness of fitness differences between
    neighbors (captures asymmetry in the local landscape).
  - Dispersion metric: std of fitness values at local optima / global range.
  - Random walk autocorrelation (already in Table 5, included for comparison).
"""

import numpy as np
import json
import sys
import os
from scipy import stats
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.orc_discrete import full_landscape_analysis, find_all_local_optima, hill_climb
from src.nk_landscape import NKLandscape
from src.wmodel import WModel


def compute_ela_features(landscape, n_samples=2000, seed=42):
    """Compute discrete ELA-style features from a landscape."""
    fitness = landscape.fitness
    neighbor_fn = landscape.neighbor_fn
    space_size = landscape.space_size
    N = landscape.N if hasattr(landscape, 'N') else landscape.n

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(space_size, size=min(n_samples, space_size), replace=False)
    sample_fitness = fitness[sample_idx]

    nbc_ratios = []
    for i, s in enumerate(sample_idx):
        nbrs = neighbor_fn(s)
        nn_dist = 1.0
        better_nbrs = [n for n in nbrs if fitness[n] < fitness[s]]
        if better_nbrs:
            nbc_ratios.append(1.0)
        else:
            for d in range(2, N+1):
                found_better = False
                visited = {s}
                frontier = set(nbrs)
                for step in range(d - 1):
                    next_frontier = set()
                    for node in frontier:
                        for nn in neighbor_fn(node):
                            if nn not in visited:
                                if fitness[nn] < fitness[s]:
                                    found_better = True
                                    break
                                next_frontier.add(nn)
                                visited.add(nn)
                        if found_better:
                            break
                    if found_better:
                        nbc_ratios.append(float(step + 2))
                        break
                    frontier = next_frontier
                if found_better:
                    break
            else:
                nbc_ratios.append(float(N))

    nbc_mean = float(np.mean(nbc_ratios))

    fc_diffs = []
    for s in sample_idx[:500]:
        nbrs = neighbor_fn(s)
        for n in nbrs:
            fc_diffs.append(fitness[n] - fitness[s])
    fc_diffs = np.array(fc_diffs)
    fc_skewness = float(stats.skew(fc_diffs)) if len(fc_diffs) > 2 else 0.0
    fc_kurtosis = float(stats.kurtosis(fc_diffs)) if len(fc_diffs) > 2 else 0.0
    fc_mean_abs = float(np.mean(np.abs(fc_diffs)))

    optima = find_all_local_optima(space_size, fitness, neighbor_fn)
    opt_fitness = fitness[optima]
    f_range = fitness.max() - fitness.min()
    dispersion = float(np.std(opt_fitness) / f_range) if f_range > 0 else 0.0
    n_optima_frac = len(optima) / space_size

    rw_length = min(1000, space_size)
    rw_fitness = []
    current = rng.randint(space_size)
    for _ in range(rw_length):
        rw_fitness.append(fitness[current])
        nbrs = neighbor_fn(current)
        current = nbrs[rng.randint(len(nbrs))]
    rw_fitness = np.array(rw_fitness)
    if np.std(rw_fitness) > 0:
        rw_autocorr = float(np.corrcoef(rw_fitness[:-1], rw_fitness[1:])[0, 1])
    else:
        rw_autocorr = 1.0

    return {
        'nbc_mean': nbc_mean,
        'fc_skewness': fc_skewness,
        'fc_kurtosis': fc_kurtosis,
        'fc_mean_abs': fc_mean_abs,
        'dispersion': dispersion,
        'n_optima_frac': n_optima_frac,
        'rw_autocorr': rw_autocorr,
    }


def analyze_config(args):
    btype, cfg, seed = args
    if btype == 'NK':
        land = NKLandscape(N=cfg['N'], K=cfg['K'], model='adjacent', seed=seed)
        label = f"NK_K{cfg['K']}"
    else:
        land = WModel(n=cfg['n'], nu=cfg['nu'], gamma=0, mu=1, seed=seed)
        label = f"W_nu{cfg['nu']}"

    features = compute_ela_features(land, seed=seed)
    features['type'] = btype
    features['config'] = cfg
    features['seed'] = seed
    features['label'] = label
    return features


def main():
    configs = [
        ('NK', {'N': 16, 'K': 4}),
        ('NK', {'N': 16, 'K': 8}),
        ('NK', {'N': 16, 'K': 12}),
        ('W',  {'n': 16, 'nu': 3}),
        ('W',  {'n': 16, 'nu': 4}),
        ('W',  {'n': 16, 'nu': 6}),
        ('W',  {'n': 16, 'nu': 8}),
    ]

    tasks = []
    for btype, cfg in configs:
        for seed in range(30):
            tasks.append((btype, cfg, seed))

    print(f"Computing ELA features for {len(tasks)} instances...")
    with Pool(min(60, len(tasks))) as pool:
        results = pool.map(analyze_config, tasks)

    ld_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'landscape_discrete_v3.json')
    with open(ld_path) as f:
        ld_data = json.load(f)

    perf_by_label_seed = {}
    for entry in ld_data:
        c = entry.get('config', {})
        N = c.get('N', c.get('n'))
        K = c.get('K', c.get('k'))
        nu = c.get('nu')
        seed = entry.get('seed', 0)
        if N != 16:
            continue
        if K is not None:
            label = f"NK_K{K}"
        elif nu is not None:
            label = f"W_nu{nu}"
        else:
            continue
        key = (label, seed)
        perf_by_label_seed[key] = {
            'hc_success': entry.get('algo_HC_success_rate', None),
            'ea_success': entry.get('algo_EA_success_rate', None),
            'rs_mean': entry.get('algo_RS_mean', None),
        }

    features_names = ['nbc_mean', 'fc_skewness', 'fc_kurtosis', 'fc_mean_abs',
                       'dispersion', 'n_optima_frac', 'rw_autocorr']
    targets = ['hc_success', 'ea_success', 'rs_mean']

    matched = []
    for r in results:
        key = (r['label'], r['seed'])
        if key in perf_by_label_seed:
            perf = perf_by_label_seed[key]
            if all(perf[t] is not None for t in targets):
                r['perf'] = perf
                matched.append(r)

    print(f"\nMatched {len(matched)} instances with performance data")

    trivial_hc = [r for r in matched if 0.01 < r['perf']['hc_success'] < 0.99]
    trivial_ea = [r for r in matched if 0.01 < r['perf']['ea_success'] < 0.99]

    print(f"\n{'='*90}")
    print(f"Spearman Correlations: ELA Features vs Algorithm Performance")
    print(f"{'='*90}")
    print(f"{'Feature':<20} {'HC success':>12} {'EA success':>12} {'RS mean':>12}")
    print(f"{'-'*60}")

    for fname in features_names:
        vals_all = [r[fname] for r in matched]
        hc_vals = [r['perf']['hc_success'] for r in matched]
        ea_vals = [r['perf']['ea_success'] for r in matched]
        rs_vals = [r['perf']['rs_mean'] for r in matched]

        rho_hc, p_hc = stats.spearmanr(vals_all, hc_vals)
        rho_ea, p_ea = stats.spearmanr(vals_all, ea_vals)
        rho_rs, p_rs = stats.spearmanr(vals_all, rs_vals)

        sig_hc = '***' if p_hc < 0.001 else '**' if p_hc < 0.01 else '*' if p_hc < 0.05 else ''
        sig_ea = '***' if p_ea < 0.001 else '**' if p_ea < 0.01 else '*' if p_ea < 0.05 else ''
        sig_rs = '***' if p_rs < 0.001 else '**' if p_rs < 0.01 else '*' if p_rs < 0.05 else ''

        print(f"{fname:<20} {rho_hc:>8.3f}{sig_hc:<4} {rho_ea:>8.3f}{sig_ea:<4} {rho_rs:>8.3f}{sig_rs:<4}")

    with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'ela_features.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/ela_features.json")


if __name__ == '__main__':
    main()
