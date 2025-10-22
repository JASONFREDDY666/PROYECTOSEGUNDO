"""
Distribuciones continuas de variables aleatorias - ACTUALIZADO
"""

import numpy as np
import math
from scipy import stats
from typing import List

class ContinuousDistributions:
    """Clase para distribuciones continuas"""
    
    @staticmethod
    def uniform(a: float, b: float, size: int) -> List[float]:
        """Distribución Uniforme U(a, b)"""
        return [round(a + (b - a) * np.random.random(), 4) for _ in range(size)]
    
    @staticmethod
    def exponential(lambd: float, size: int) -> List[float]:
        """Distribución Exponencial E(λ)"""
        return [round(-math.log(1 - np.random.random()) / lambd, 4) for _ in range(size)]
    
    @staticmethod
    def normal(mu: float, sigma: float, size: int) -> List[float]:
        """Distribución Normal N(μ, σ)"""
        return [round(np.random.normal(mu, sigma), 4) for _ in range(size)]
    
    @staticmethod
    def gamma(alpha: float, beta: float, size: int) -> List[float]:
        """Distribución Gamma Γ(α, β)"""
        return [round(np.random.gamma(alpha, beta), 4) for _ in range(size)]
    
    @staticmethod
    def weibull(alpha: float, beta: float, size: int) -> List[float]:
        """Distribución Weibull W(α, β)"""
        return [round(beta * (-math.log(1 - np.random.random())) ** (1/alpha), 4) for _ in range(size)]
    
    @staticmethod
    def erlang(k: int, lambd: float, size: int) -> List[float]:
        """Distribución K-Erlang"""
        # K-Erlang es un caso especial de Gamma con α=k y β=1/λ
        return [round(np.random.gamma(k, 1/lambd), 4) for _ in range(size)]