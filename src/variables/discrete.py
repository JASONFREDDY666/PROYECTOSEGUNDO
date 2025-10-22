"""
Distribuciones discretas de variables aleatorias
"""

import numpy as np
import math
from typing import List

class DiscreteDistributions:
    """Clase para distribuciones discretas"""
    
    @staticmethod
    def uniform(a: int, b: int, size: int) -> List[int]:
        """Distribución Uniforme Discreta U(a, b)"""
        return [np.random.randint(a, b + 1) for _ in range(size)]
    
    @staticmethod
    def bernoulli(p: float, size: int) -> List[int]:
        """Distribución Bernoulli Bern(p)"""
        return [1 if np.random.random() < p else 0 for _ in range(size)]
    
    @staticmethod
    def binomial(n: int, p: float, size: int) -> List[int]:
        """Distribución Binomial B(n, p)"""
        return [np.random.binomial(n, p) for _ in range(size)]
    
    @staticmethod
    def poisson(lambd: float, size: int) -> List[int]:
        """Distribución Poisson P(λ)"""
        return [np.random.poisson(lambd) for _ in range(size)]
    
    @staticmethod
    def geometric(p: float, size: int) -> List[int]:
        """Distribución Geométrica Geo(p)"""
        return [np.random.geometric(p) for _ in range(size)]

class DiscreteTheoretical:
    """Cálculos teóricos para distribuciones discretas"""
    
    @staticmethod
    def uniform_pmf(x: int, a: int, b: int) -> float:
        """Función de masa de probabilidad Uniforme"""
        return 1/(b - a + 1) if a <= x <= b else 0
    
    @staticmethod
    def bernoulli_pmf(x: int, p: float) -> float:
        """Función de masa de probabilidad Bernoulli"""
        return p if x == 1 else (1-p) if x == 0 else 0
    
    @staticmethod
    def binomial_pmf(x: int, n: int, p: float) -> float:
        """Función de masa de probabilidad Binomial"""
        if x < 0 or x > n:
            return 0
        return math.comb(n, x) * (p ** x) * ((1-p) ** (n-x))
    
    @staticmethod
    def poisson_pmf(x: int, lambd: float) -> float:
        """Función de masa de probabilidad Poisson"""
        if x < 0:
            return 0
        return (math.exp(-lambd) * (lambd ** x)) / math.factorial(x)
    
    @staticmethod
    def get_theoretical_values(distribution: str, params: dict, x_range: np.ndarray) -> np.ndarray:
        """Obtiene valores teóricos de la PMF"""
        if distribution == "uniform":
            a, b = params['a'], params['b']
            return np.array([DiscreteTheoretical.uniform_pmf(int(x), a, b) for x in x_range])
        elif distribution == "bernoulli":
            p = params['p']
            return np.array([DiscreteTheoretical.bernoulli_pmf(int(x), p) for x in x_range])
        elif distribution == "binomial":
            n, p = params['n'], params['p']
            return np.array([DiscreteTheoretical.binomial_pmf(int(x), n, p) for x in x_range])
        elif distribution == "poisson":
            lambd = params['lambda']
            return np.array([DiscreteTheoretical.poisson_pmf(int(x), lambd) for x in x_range])
        else:
            return np.zeros_like(x_range)