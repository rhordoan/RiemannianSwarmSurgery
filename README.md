# Geometric Deep Optimization: Riemannian Swarm Surgery (RSS)

## 1. The Paradigm Shift: From Traversal to Surgery
Prevailing optimization paradigms (gradient descent, CMA-ES) treat the objective function as a **static, immutable landscape**. The optimizer acts as an explorer navigating a fixed Riemannian manifold $(M, g)$. In high dimensions ($D > 100$) and complex "Russian Doll" topologies (like IEEE CEC 2022 Composition Functions), this assumption fails. The topology itself—characterized by hyperbolic necks and non-contractible loops—conspires against the agent.

**Riemannian Swarm Surgery (RSS)** proposes a foundational shift to **Geometric Deep Optimization (GDO)**. Instead of merely searching the manifold, we dynamically **evolve the manifold's metric tensor** $g_{ij}$ to simplify the problem structure itself.

> "The algorithm does not simply climb the hill; it deforms the hill until it becomes a plateau or a single peak."

## 2. Core Mechanisms

RSS integrates three advanced mathematical frameworks to perform this surgery:

### 2.1 Discrete Forman-Ricci Flow (The Engine)
We utilize a combinatorial discretization of Ricci curvature (Forman-Ricci Curvature) on the swarm's $k$-Nearest Neighbor graph.
*   **Positive Curvature (Basins)**: The flow contracts these regions, creating an attractive "gravity well" that accelerates convergence.
*   **Negative Curvature (Bottlenecks/Saddles)**: The flow expands these regions, pushing agents apart and widening narrow passages.
This allows the algorithm to actively reshape the optimization landscape in $O(1)$ time per edge.

### 2.2 Topological Scouting via Persistent Homology (The Scout)
While Ricci flow acts locally, we need a global "Topological Eye". We use **Persistent Homology** (via Vietoris-Rips filtration) to compute Betti numbers:
*   **$\beta_0$ (Connected Components)**: Detects distinct basins of attraction.
*   **$\beta_1$ (Cycles/Loops)**: Detects when agents are trapped circumnavigating a central peak or spiral ridge (common in Composition Functions).
This allows the system to detect global topological obstructions that local gradients cannot see.

### 2.3 Topological Surgery (The "Cut and Cap")
When a topological obstruction is confirmed (e.g., a persistent $\beta_1$ loop or a "dumbell" shape), the algorithm performs surgery:
1.  **Cut**: Edges with highly negative curvature are severed, topologically splitting the swarm into independent sub-species.
2.  **Cap**: The local metric is re-stabilized, allowing sub-swarms to optimize independently without bad gradient interference.

### 2.4 Sheaf-Theoretic Memory (The Archive)
To prevent the swarm from cycling back into previously pruned regions, we model the search history as a **Cellular Sheaf**. "Ghost Topologies" of failed or explored regions are stored, and new agents act as sections that are repelled from these regions via metric inflation if they show local consistency with the ghosts.

## 3. Goals
The primary objective of this framework is to solve the "**Russian Doll**" traps found in the **IEEE CEC 2022 Composition Functions (F11 & F12)**, where standard state-of-the-art algorithms (like L-SHADE variants and CMA-ES) stagnate due to inability to handle conflicting curvature signals between nested basins.

## 4. Usage

### Installation
```bash
pip install -r requirements.txt
# Optional: Install fast C++ curvature library
# pip install GraphRicciCurvature
```

### Running the Optimizer
```python
from benchmarks.rss_optimizer import RSSOptimizer
from benchmarks.synthetic_functions import RussianDollFunction

problem = RussianDollFunction(dimension=10)
opt = RSSOptimizer(problem, pop_size=30, dim=10, max_fe=20000, archive_type='sheaf')
history = opt.run()
```

### Reproducing Benchmark
```bash
python benchmarks/run_benchmark.py
```
Results will be saved to `results/convergence_d10.png`.

## 5. Preliminary Results (Synthetic Russian Doll)
RSS-Sheaf demonstrates robust convergence behavior compared to baseline methods. The topological surgery mechanism correctly identifies and severs connections when the swarm becomes entangled in the "Russian Doll" nested basins, preventing the cyclic stagnation observed in standard CMA-ES.

## 6. Architecture
