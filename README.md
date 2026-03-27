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
│   ├── landscape_metrics.py    # Classical FLA metrics (FDC, autocorrelation, ELA)
│   └── ollivier_ricci.py       # Base Ollivier-Ricci implementation
│
├── benchmarks/                 # Experiment scripts (paper results)
│   ├── landscape_analysis_discrete.py   # Main escape rate & correlation analysis
│   ├── otg_analysis.py                  # OTG vs LON funnel detection
│   ├── orc_ils.py                       # ILS comparison experiments
│   ├── maxsat_otg.py                    # MAX-SAT scaling experiments
│   ├── qap_otg.py                       # QAP experiments
│   ├── tsp_2opt_experiment.py           # TSP 2-opt experiments
│   ├── tsp_2opt_scaling.py              # TSP scaling analysis
│   └── ela_features.py                  # ELA feature comparison
│
├── results/                    # Experimental results (JSON/CSV)
│   ├── landscape_discrete_v3.json       # NK/W-model escape rates (N=16)
│   ├── landscape_discrete_n20.json      # NK escape rates (N=20)
│   ├── otg_analysis_v4.json             # OTG vs LON funnel quality
│   ├── orc_ils_v3.json                  # ILS comparison results
│   ├── maxsat_otg_scaling.json          # MAX-SAT results (N=20,50,100)
│   ├── qap_otg.json                     # QAP results
│   ├── tsp_2opt_results.json            # TSP enumeration results
│   └── tsp_2opt_scaling.json            # TSP scaling results
│
├── paper/                      # LaTeX source (Springer LNCS)
│   ├── main.tex                # Paper source
│   ├── llncs.cls               # LNCS document class
│   └── figures/                # Generated figures
│
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, NetworkX, Matplotlib, scikit-learn, IOHexperimenter.

## Reproducing Paper Results

Each experiment script in `benchmarks/` is self-contained and writes results to `results/`.

### Table 1 & 6: Escape rates and correlation analysis (N=16)
```bash
python benchmarks/landscape_analysis_discrete.py
```

### Table 2: OTG vs LON funnel quality
```bash
python benchmarks/otg_analysis.py
```

### Table 3: MAX-SAT scaling (N=20, 50, 100)
```bash
python benchmarks/maxsat_otg.py
```

### Table 4: QAP experiments
```bash
python benchmarks/qap_otg.py
```

### Table 5: TSP 2-opt experiments
```bash
python benchmarks/tsp_2opt_experiment.py    # Full enumeration (n=8,9,10)
python benchmarks/tsp_2opt_scaling.py       # Scaling (n=15,20,30,50)
```

### Table 6 (ILS comparison)
```bash
python benchmarks/orc_ils.py
```

### Table 7: N=20 escape rates
```bash
python benchmarks/landscape_analysis_discrete.py  # Includes N=20 configurations
```

### Figures
```bash
python paper/generate_new_figures.py
```

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
