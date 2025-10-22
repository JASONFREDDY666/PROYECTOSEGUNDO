"""
Módulo de generadores de números pseudoaleatorios
"""

import numpy as np
import time
from typing import List, Tuple

class RandomGenerator:
    """Clase base para generadores de números pseudoaleatorios"""
    
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else int(time.time())
        self.current_value = self.seed
        self.generated_numbers = []
        
    def reset(self, seed=None):
        """Reinicia el generador con una nueva semilla"""
        if seed is not None:
            self.seed = seed
        self.current_value = self.seed
        self.generated_numbers = []
        
    def generate(self, n: int) -> List[float]:
        """Genera n números pseudoaleatorios entre 0 y 1"""
        raise NotImplementedError("Método debe ser implementado por subclases")

class LinearCongruentialGenerator(RandomGenerator):
    """Generador Congruencial Lineal"""
    
    def __init__(self, seed=None, a=1664525, c=1013904223, m=2**32):
        super().__init__(seed)
        self.a = a  # Multiplicador
        self.c = c  # Incremento
        self.m = m  # Módulo
        
    def generate(self, n: int) -> List[float]:
        """Genera n números usando el método congruencial lineal"""
        numbers = []
        for _ in range(n):
            self.current_value = (self.a * self.current_value + self.c) % self.m
            random_num = self.current_value / self.m
            numbers.append(round(random_num, 4))
        self.generated_numbers.extend(numbers)
        return numbers

class MiddleSquareGenerator(RandomGenerator):
    """Generador de Cuadrados Medios"""
    
    def __init__(self, seed=None, digits=4):
        super().__init__(seed)
        self.digits = digits
        self.modulus = 10 ** digits
        
    def generate(self, n: int) -> List[float]:
        """Genera n números usando el método de cuadrados medios"""
        numbers = []
        current = self.seed
        
        for _ in range(n):
            # Asegurar que el número tenga la cantidad correcta de dígitos
            current = int(str(current).zfill(self.digits)[-self.digits:])
            
            # Elevar al cuadrado
            squared = current ** 2
            
            # Tomar los dígitos del medio
            squared_str = str(squared).zfill(self.digits * 2)
            start = (len(squared_str) - self.digits) // 2
            middle_digits = squared_str[start:start + self.digits]
            
            current = int(middle_digits) if middle_digits else 0
            random_num = current / self.modulus
            numbers.append(round(random_num, 4))
            
        self.generated_numbers.extend(numbers)
        return numbers

class FibonacciGenerator(RandomGenerator):
    """Generador de Fibonacci"""
    
    def __init__(self, seed1=None, seed2=None, m=2**32):
        seed1 = seed1 if seed1 is not None else int(time.time())
        seed2 = seed2 if seed2 is not None else int(time.time() * 0.5)
        super().__init__(seed1)
        self.seed2 = seed2
        self.prev_value = seed2
        self.m = m
        
    def generate(self, n: int) -> List[float]:
        """Genera n números usando el método de Fibonacci"""
        numbers = []
        x_n_minus_1 = self.seed
        x_n_minus_2 = self.seed2
        
        for _ in range(n):
            x_n = (x_n_minus_1 + x_n_minus_2) % self.m
            random_num = x_n / self.m
            numbers.append(round(random_num, 4))
            
            x_n_minus_2 = x_n_minus_1
            x_n_minus_1 = x_n
            
        self.generated_numbers.extend(numbers)
        return numbers

class UniformDistributionGenerator:
    """Generador de distribución uniforme"""
    
    @staticmethod
    def generate(a: float, b: float, n: int, generator: RandomGenerator) -> List[float]:
        """Genera números con distribución uniforme U(a, b)"""
        uniform_numbers = generator.generate(n)
        return [round(a + (b - a) * num, 4) for num in uniform_numbers]

# Fábrica de generadores
class GeneratorFactory:
    """Fábrica para crear diferentes tipos de generadores"""
    
    @staticmethod
    def create_generator(generator_type: str, **kwargs) -> RandomGenerator:
        """Crea un generador basado en el tipo especificado"""
        if generator_type == "congruential":
            return LinearCongruentialGenerator(**kwargs)
        elif generator_type == "middle_square":
            return MiddleSquareGenerator(**kwargs)
        elif generator_type == "fibonacci":
            return FibonacciGenerator(**kwargs)
        else:
            raise ValueError(f"Tipo de generador no soportado: {generator_type}")

# Pruebas de calidad básicas
class GeneratorTests:
    """Pruebas básicas de calidad para generadores"""
    
    @staticmethod
    def test_uniformity(numbers: List[float], num_intervals: int = 10) -> dict:
        """Prueba de uniformidad usando frecuencia en intervalos"""
        if not numbers:
            return {"error": "No hay números para probar"}
            
        intervals = [0] * num_intervals
        interval_size = 1.0 / num_intervals
        
        for num in numbers:
            if 0 <= num <= 1:
                index = min(int(num / interval_size), num_intervals - 1)
                intervals[index] += 1
        
        expected = len(numbers) / num_intervals
        chi_square = sum((observed - expected) ** 2 / expected for observed in intervals)
        
        return {
            "intervals": intervals,
            "expected_per_interval": expected,
            "chi_square": round(chi_square, 4),
            "is_uniform": chi_square < 16.92  # Para 9 grados de libertad, alpha=0.05
        }
    
    @staticmethod
    def test_independence(numbers: List[float]) -> dict:
        """Prueba básica de independencia usando autocorrelación"""
        if len(numbers) < 2:
            return {"error": "Se necesitan al menos 2 números"}
            
        # Autocorrelación en lag 1
        mean = np.mean(numbers)
        variance = np.var(numbers)
        
        if variance == 0:
            return {"autocorrelation": 0, "is_independent": True}
            
        autocorr = np.corrcoef(numbers[:-1], numbers[1:])[0, 1]
        
        return {
            "autocorrelation": round(autocorr, 4),
            "is_independent": abs(autocorr) < 0.1  # Umbral arbitrario
        }