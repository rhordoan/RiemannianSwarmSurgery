# ORC Transition Graph: Ollivier–Ricci Curvature for Discrete Fitness Landscape Analysis

Companion code and data for the paper:

> **Ollivier–Ricci Curvature as a Landscape Analysis Framework for Discrete Combinatorial Optimization**
> Roberto Sergiu Hordoan — Babeș–Bolyai University, Cluj-Napoca, Romania
> *Submitted to PPSN 2026 (Springer LNCS)*

## Overview

This repository implements the **ORC Transition Graph (OTG)**, a deterministic directed graph over local optima derived from Ollivier–Ricci curvature (ORC). The OTG extracts cross-basin structural information from fitness landscapes using only 1-hop neighborhood data, achieving up to 9.2× better attractor quality than LON-d1 on epistatic W-model landscapes.

The framework covers three foundational neighborhoods in combinatorial optimization:
- **Binary hypercubes** (bit-flip) — NK landscapes, W-model, MAX-SAT
- **Permutation swap graphs** — QAP
- **TSP 2-opt neighborhoods** — TSP

## Repository Structure

```
├── src/                        # Core library
│   ├── orc_discrete.py         # Generic discrete ORC computation
│   ├── orc_tsp.py              # ORC for TSP 2-opt neighborhoods
│   ├── nk_landscape.py         # NK landscape generator
│   ├── wmodel.py               # W-model benchmark generator
│   └── landscape_metrics.py    # Classical FLA metrics (FDC, autocorrelation, ELA)
│
├── benchmarks/                 # Experiment scripts (reproduce paper results)
│   ├── landscape_analysis_discrete.py   # Escape rate & correlation analysis
│   ├── otg_analysis.py                  # OTG vs LON funnel detection
│   ├── orc_ils.py                       # ILS comparison experiments
│   ├── maxsat_otg.py                    # MAX-SAT experiments
│   ├── maxsat_otg_scaling.py            # MAX-SAT scaling (N=20,50,100)
│   ├── qap_otg.py                       # QAP experiments
│   ├── tsp_2opt_experiment.py           # TSP 2-opt experiments
│   ├── tsp_2opt_scaling.py              # TSP scaling analysis
│   ├── ela_features.py                  # ELA feature comparison
│   └── within_k_analysis.py             # Within-K correlation analysis
│
├── results/                    # Pre-computed experimental results (JSON)
├── paper/                      # LaTeX source (Springer LNCS)
├── archive/                    # Earlier continuous optimization experiments (not used in paper)
└── requirements.txt            # Python dependencies
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick smoke test — runs a small NK landscape (~10 seconds)
python3 -c "
from src.nk_landscape import NKLandscape
from src.orc_discrete import full_landscape_analysis
nk = NKLandscape(N=10, K=2, seed=42)
result = full_landscape_analysis(nk.space_size, nk.fitness, nk.neighbor_fn)
print(f'Search space: 2^10 = {nk.space_size} solutions')
print(f'Local optima: {result[\"n_local_optima\"]}')
print(f'ORC escape success: {result[\"frac_leads_to_better\"]:.1%}')
print('Setup OK!')
"

# 3. Run a fast experiment (NK landscape, N=16, K=4 — ~2 minutes)
python3 benchmarks/landscape_analysis_discrete.py --N 16 --K 4 --instances 2
```

## Reproducing Paper Results

Each benchmark script writes results to `results/`. Pre-computed outputs are
included so figures can be regenerated without re-running experiments.

| Paper Section | Command | Runtime |
|---|---|---|
| Tables 1 & 6 — Escape rates (N=16) | `python benchmarks/landscape_analysis_discrete.py` | ~30 min |
| Table 2 — OTG vs LON funnels | `python benchmarks/otg_analysis.py` | ~20 min |
| Table 3 — MAX-SAT scaling | `python benchmarks/maxsat_otg.py` | ~15 min |
| Table 4 — QAP | `python benchmarks/qap_otg.py` | ~10 min |
| Table 5 — TSP 2-opt | `python benchmarks/tsp_2opt_experiment.py` | ~10 min |
| Table 5 — TSP scaling | `python benchmarks/tsp_2opt_scaling.py` | ~20 min |
| ILS comparison | `python benchmarks/orc_ils.py` | ~15 min |
| Figures 1 & 2 | `python paper/generate_new_figures.py` | ~1 min |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{hordoan2026orc,
  title     = {Ollivier--Ricci Curvature as a Landscape Analysis Framework
               for Discrete Combinatorial Optimization},
  author    = {Hordoan, Roberto Sergiu},
  booktitle = {Parallel Problem Solving from Nature (PPSN 2026)},
  series    = {LNCS},
  publisher = {Springer},
  year      = {2026}
}
```

## License

This project is provided for academic reproducibility.
