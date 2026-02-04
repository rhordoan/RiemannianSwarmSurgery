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
    - Outer: Wide, deceptive Schwefel-like or Sphere
    - Middle: Rotated Griewank (Ridges)
    - Inner: Rastrigin (Egg carton)
    
    For simplicity, we blend them radially.
    """
    def __init__(self, dimension=10):
        self.dim = dimension
        self.bounds = [-100, 100]
        
    def evaluate(self, x):
        r = np.linalg.norm(x)
        
        # Inner Basin (Radius < 5): Rastrigin (Global Optimum at 0)
        if r < 5.0:
            return rastrigin(x)
        
        # Neck/Barrier (5 < Radius < 20): High wall / Ridge
        elif r < 15.0:
            # Create a ridge that strictly increases then decreases?
            # Or just a high value chaos.
            return 200.0 + griewank(x) * 10
            
        # Outer Basin (Radius > 20): Sphere/Schwefel pointing to local optima diverse from 0
        else:
            # Bias towards a deceptive local optimum at (50, 50, ...)
            deceptive_center = np.ones_like(x) * 50
            dist_deceptive = np.linalg.norm(x - deceptive_center)
            return 100.0 + dist_deceptive**2 * 0.1
