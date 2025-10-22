"""
Módulo de pruebas estadísticas para validar generadores
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import math

class StatisticalTests:
    """Clase para realizar pruebas estadísticas de calidad"""
    
    @staticmethod
    def chi_square_test(numbers: List[float], intervals: int = 10, alpha: float = 0.05) -> Dict:
        """
        Prueba de Chi-cuadrado para uniformidad
        
        Args:
            numbers: Lista de números a probar
            intervals: Número de intervalos
            alpha: Nivel de significancia
            
        Returns:
            Dict con resultados de la prueba
        """
        if not numbers:
            return {"error": "No hay números para probar"}
        
        n = len(numbers)
        expected_frequency = n / intervals
        observed_frequencies = [0] * intervals
        
        # Contar frecuencias observadas
        for num in numbers:
            if 0 <= num <= 1:
                index = min(int(num * intervals), intervals - 1)
                observed_frequencies[index] += 1
        
        # Calcular estadístico chi-cuadrado
        chi_square = 0
        for observed in observed_frequencies:
            chi_square += (observed - expected_frequency) ** 2 / expected_frequency
        
        # Grados de libertad
        degrees_of_freedom = intervals - 1
        
        # Valor crítico
        critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
        
        # Decisión
        is_uniform = chi_square <= critical_value
        
        return {
            "test_name": "Chi-cuadrado de Uniformidad",
            "chi_square_statistic": round(chi_square, 4),
            "critical_value": round(critical_value, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "alpha": alpha,
            "is_uniform": is_uniform,
            "observed_frequencies": observed_frequencies,
            "expected_frequency": expected_frequency,
            "intervals": intervals
        }
    
    @staticmethod
    def kolmogorov_smirnov_test(numbers: List[float], alpha: float = 0.05) -> Dict:
        """
        Prueba de Kolmogorov-Smirnov para uniformidad
        
        Args:
            numbers: Lista de números a probar
            alpha: Nivel de significancia
            
        Returns:
            Dict con resultados de la prueba
        """
        if not numbers:
            return {"error": "No hay números para probar"}
        
        n = len(numbers)
        sorted_numbers = sorted(numbers)
        
        # Calcular D+ y D-
        d_plus = 0
        d_minus = 0
        
        for i in range(n):
            f_n = (i + 1) / n
            f_x = sorted_numbers[i]
            d_plus = max(d_plus, f_n - f_x)
            d_minus = max(d_minus, f_x - i / n)
        
        d_statistic = max(d_plus, d_minus)
        
        # Valor crítico de Kolmogorov-Smirnov
        critical_value = 1.36 / math.sqrt(n)  # Aproximación para alpha=0.05
        
        # Decisión
        is_uniform = d_statistic <= critical_value
        
        return {
            "test_name": "Kolmogorov-Smirnov",
            "d_statistic": round(d_statistic, 4),
            "critical_value": round(critical_value, 4),
            "d_plus": round(d_plus, 4),
            "d_minus": round(d_minus, 4),
            "alpha": alpha,
            "is_uniform": is_uniform,
            "sample_size": n
        }
    
    @staticmethod
    def runs_test(numbers: List[float], alpha: float = 0.05) -> Dict:
        """
        Prueba de Rachas (Runs Test) para independencia
        
        Args:
            numbers: Lista de números a probar
            alpha: Nivel de significancia
            
        Returns:
            Dict con resultados de la prueba
        """
        if not numbers:
            return {"error": "No hay números para probar"}
        
        n = len(numbers)
        median = np.median(numbers)
        
        # Crear secuencia de signos
        signs = ['+' if x >= median else '-' for x in numbers]
        
        # Contar rachas
        runs = 1
        for i in range(1, n):
            if signs[i] != signs[i-1]:
                runs += 1
        
        # Estadístico esperado y varianza
        n1 = signs.count('+')
        n2 = signs.count('-')
        
        expected_runs = (2 * n1 * n2) / n + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))
        
        if variance_runs == 0:
            z_statistic = 0
        else:
            z_statistic = (runs - expected_runs) / math.sqrt(variance_runs)
        
        # Valor crítico (distribución normal)
        critical_value = stats.norm.ppf(1 - alpha/2)
        
        # Decisión
        is_independent = abs(z_statistic) <= critical_value
        
        return {
            "test_name": "Prueba de Rachas",
            "runs_count": runs,
            "expected_runs": round(expected_runs, 4),
            "z_statistic": round(z_statistic, 4),
            "critical_value": round(critical_value, 4),
            "alpha": alpha,
            "is_independent": is_independent,
            "n1": n1,
            "n2": n2,
            "median": round(median, 4)
        }
    
    @staticmethod
    def autocorrelation_test(numbers: List[float], lag: int = 1, alpha: float = 0.05) -> Dict:
        """
        Prueba de autocorrelación para independencia
        
        Args:
            numbers: Lista de números a probar
            lag: Desfase para la autocorrelación
            alpha: Nivel de significancia
            
        Returns:
            Dict con resultados de la prueba
        """
        if not numbers or len(numbers) <= lag:
            return {"error": "No hay suficientes números para probar"}
        
        n = len(numbers)
        
        # Calcular autocorrelación
        mean = np.mean(numbers)
        numerator = 0
        denominator = 0
        
        for i in range(n - lag):
            numerator += (numbers[i] - mean) * (numbers[i + lag] - mean)
            denominator += (numbers[i] - mean) ** 2
        
        for i in range(n - lag, n):
            denominator += (numbers[i] - mean) ** 2
        
        autocorrelation = numerator / denominator if denominator != 0 else 0
        
        # Estadístico de prueba
        z_statistic = autocorrelation * math.sqrt(n - lag)
        
        # Valor crítico (distribución normal)
        critical_value = stats.norm.ppf(1 - alpha/2)
        
        # Decisión
        is_independent = abs(z_statistic) <= critical_value
        
        return {
            "test_name": "Prueba de Autocorrelación",
            "autocorrelation": round(autocorrelation, 4),
            "lag": lag,
            "z_statistic": round(z_statistic, 4),
            "critical_value": round(critical_value, 4),
            "alpha": alpha,
            "is_independent": is_independent,
            "sample_size": n
        }
    
    @staticmethod
    def poker_test(numbers: List[float], alpha: float = 0.05) -> Dict:
        """
        Prueba de Poker para patrones en dígitos
        
        Args:
            numbers: Lista de números a probar
            alpha: Nivel de significancia
            
        Returns:
            Dict con resultados de la prueba
        """
        if not numbers:
            return {"error": "No hay números para probar"}
        
        # Convertir números a strings de 4 dígitos
        digit_strings = [f"{num:.4f}"[2:6] for num in numbers]
        
        # Categorías del poker test
        categories = {
            "Todos diferentes": 0,  # ABCD
            "Un par": 0,            # AABC
            "Dos pares": 0,         # AABB
            "Tercia": 0,            # AAAB
            "Poker": 0,             # AAAA
            "Full house": 0         # No aplica para 4 dígitos
        }
        
        # Contar patrones
        for digits in digit_strings:
            digit_counts = {}
            for digit in digits:
                digit_counts[digit] = digit_counts.get(digit, 0) + 1
            
            counts = sorted(digit_counts.values(), reverse=True)
            
            if counts == [4]:
                categories["Poker"] += 1
            elif counts == [3, 1]:
                categories["Tercia"] += 1
            elif counts == [2, 2]:
                categories["Dos pares"] += 1
            elif counts == [2, 1, 1]:
                categories["Un par"] += 1
            elif counts == [1, 1, 1, 1]:
                categories["Todos diferentes"] += 1
        
        # Frecuencias esperadas (para 4 dígitos)
        n = len(numbers)
        expected = {
            "Todos diferentes": n * 0.504,
            "Un par": n * 0.432,
            "Dos pares": n * 0.027,
            "Tercia": n * 0.036,
            "Poker": n * 0.001
        }
        
        # Calcular chi-cuadrado
        chi_square = 0
        for category in expected:
            if expected[category] > 0:  # Evitar división por cero
                observed = categories.get(category, 0)
                chi_square += (observed - expected[category]) ** 2 / expected[category]
        
        # Grados de libertad (5 categorías - 1)
        degrees_of_freedom = 4
        critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
        
        is_random = chi_square <= critical_value
        
        return {
            "test_name": "Prueba de Poker",
            "chi_square_statistic": round(chi_square, 4),
            "critical_value": round(critical_value, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "alpha": alpha,
            "is_random": is_random,
            "observed_frequencies": categories,
            "expected_frequencies": {k: round(v, 2) for k, v in expected.items()}
        }

class TestSuite:
    """Suite completa de pruebas estadísticas"""
    
    def __init__(self):
        self.tests = StatisticalTests()
    
    def run_full_suite(self, numbers: List[float]) -> Dict:
        """
        Ejecuta todas las pruebas estadísticas
        
        Args:
            numbers: Lista de números a probar
            
        Returns:
            Dict con resultados de todas las pruebas
        """
        results = {}
        
        # Pruebas de uniformidad
        results["chi_square"] = self.tests.chi_square_test(numbers)
        results["kolmogorov_smirnov"] = self.tests.kolmogorov_smirnov_test(numbers)
        
        # Pruebas de independencia
        results["runs_test"] = self.tests.runs_test(numbers)
        results["autocorrelation"] = self.tests.autocorrelation_test(numbers)
        results["poker_test"] = self.tests.poker_test(numbers)
        
        # Resumen general
        uniformity_tests = [results["chi_square"].get("is_uniform", False),
                          results["kolmogorov_smirnov"].get("is_uniform", False)]
        
        independence_tests = [results["runs_test"].get("is_independent", False),
                            results["autocorrelation"].get("is_independent", False),
                            results["poker_test"].get("is_random", False)]
        
        uniformity_score = sum(uniformity_tests) / len(uniformity_tests)
        independence_score = sum(independence_tests) / len(independence_tests)
        overall_score = (uniformity_score + independence_score) / 2
        
        results["summary"] = {
            "uniformity_score": round(uniformity_score * 100, 2),
            "independence_score": round(independence_score * 100, 2),
            "overall_score": round(overall_score * 100, 2),
            "is_acceptable": overall_score >= 0.7
        }
        
        return results