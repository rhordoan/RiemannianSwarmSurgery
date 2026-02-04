import numpy as np

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewank(x):
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def sphere(x):
    return np.sum(x**2)

class RussianDollFunction:
    """
    Synthetic 'Russian Doll' function mimicking CEC 2022 F12.
    Nested basins with conflicting curvature.
    
    Structure:
    - Inner: Global optimum at origin (Sphere-like)
    - Barrier: Ring-shaped ridge creating a "moat"
    - Outer: Deceptive basin pulling towards local optimum
    
    Uses smooth blending to ensure gradients exist everywhere.
    """
    def __init__(self, dimension=10):
        self.dim = dimension
        self.bounds = [-100, 100]
        
    def evaluate(self, x):
        r = np.linalg.norm(x)
        
        # Inner Basin: Global optimum at 0
        # f_inner(x) = ||x||^2 (Sphere)
        f_inner = np.sum(x**2)
        
        # Barrier: Ring-shaped ridge at radius ~10
        # Gaussian bump centered at r=10 with width 5
        barrier_center = 10.0
        barrier_width = 5.0
        barrier_height = 500.0
        f_barrier = barrier_height * np.exp(-((r - barrier_center)**2) / (2 * barrier_width**2))
        
        # Outer Basin: Deceptive attractor at (30, 30, ..., 30)
        deceptive_center = np.ones_like(x) * 30.0
        f_outer = 100.0 + 0.5 * np.sum((x - deceptive_center)**2) / self.dim
        
        # Smooth blending using sigmoid weights based on radius
        # Inner dominates when r < 5, outer dominates when r > 20
        w_inner = 1.0 / (1.0 + np.exp((r - 5) / 2))  # Sigmoid centered at r=5
        w_outer = 1.0 / (1.0 + np.exp(-(r - 20) / 2))  # Sigmoid centered at r=20
        w_barrier = 1.0 - w_inner - w_outer
        w_barrier = max(0, w_barrier)  # Clamp
        
        # Weighted combination
        f = w_inner * f_inner + w_barrier * f_barrier + w_outer * f_outer
        
        return f
