"""
Módulo para pruebas de Chi-cuadrado y análisis de bondad de ajuste
Implementa pruebas estadísticas para distribuciones uniformes y personalizadas
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Union, Tuple
import math

class ChiSquareTest:
    """
    Clase para realizar pruebas de Chi-cuadrado de bondad de ajuste
    """
    
    def __init__(self):
        self.results = {}
        self.critical_values = {
            0.01: {
                1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086,
                6: 16.812, 7: 18.475, 8: 20.090, 9: 21.666, 10: 23.209,
                11: 24.725, 12: 26.217, 13: 27.688, 14: 29.141, 15: 30.578,
                16: 32.000, 17: 33.409, 18: 34.805, 19: 36.191, 20: 37.566
            },
            0.05: {
                1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
                6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
                11: 19.675, 12: 21.026, 13: 22.362, 14: 23.685, 15: 24.996,
                16: 26.296, 17: 27.587, 18: 28.869, 19: 30.144, 20: 31.410
            },
            0.10: {
                1: 2.706, 2: 4.605, 3: 6.251, 4: 7.779, 5: 9.236,
                6: 10.645, 7: 12.017, 8: 13.362, 9: 14.684, 10: 15.987,
                11: 17.275, 12: 18.549, 13: 19.812, 14: 21.064, 15: 22.307,
                16: 23.542, 17: 24.769, 18: 25.989, 19: 27.204, 20: 28.412
            }
        }
    
    def test_uniformity(self, numbers: List[float], intervals: int = 10, 
                       alpha: float = 0.05) -> Dict:
        """
        Prueba de Chi-cuadrado para uniformidad
        
        Args:
            numbers: Lista de números entre 0 y 1
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
        
        # Calcular estadístico Chi-cuadrado
        chi_square = 0
        for observed in observed_frequencies:
            chi_square += (observed - expected_frequency) ** 2 / expected_frequency
        
        # Grados de libertad
        degrees_of_freedom = intervals - 1
        
        # Valor crítico
        critical_value = self._get_critical_value(degrees_of_freedom, alpha)
        
        # Valor p
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)
        
        # Decisión
        is_uniform = chi_square <= critical_value
        
        self.results = {
            "test_name": "Chi-cuadrado de Uniformidad",
            "sample_size": n,
            "intervals": intervals,
            "chi_square_statistic": round(chi_square, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "critical_value": round(critical_value, 4),
            "p_value": round(p_value, 6),
            "alpha": alpha,
            "is_uniform": is_uniform,
            "observed_frequencies": observed_frequencies,
            "expected_frequency": expected_frequency,
            "decision": "No rechazar H0 (Uniforme)" if is_uniform else "Rechazar H0 (No uniforme)"
        }
        
        return self.results
    
    def test_custom_distribution(self, observed: List[float], 
                                expected: List[float], 
                                alpha: float = 0.05) -> Dict:
        """
        Prueba de Chi-cuadrado para distribución personalizada
        
        Args:
            observed: Frecuencias observadas
            expected: Frecuencias esperadas
            alpha: Nivel de significancia
        
        Returns:
            Dict con resultados de la prueba
        """
        if len(observed) != len(expected):
            return {"error": "Las listas observadas y esperadas deben tener la misma longitud"}
        
        if sum(observed) == 0:
            return {"error": "Las frecuencias observadas no pueden ser todas cero"}
        
        # Ajustar frecuencias esperadas si son muy pequeñas
        adjusted_expected = self._adjust_expected_frequencies(expected, min_frequency=5)
        
        # Calcular estadístico Chi-cuadrado
        chi_square = 0
        for obs, exp in zip(observed, adjusted_expected):
            if exp > 0:  # Evitar división por cero
                chi_square += (obs - exp) ** 2 / exp
        
        # Grados de libertad
        degrees_of_freedom = len(observed) - 1
        
        # Valor crítico
        critical_value = self._get_critical_value(degrees_of_freedom, alpha)
        
        # Valor p
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)
        
        # Decisión
        fits_distribution = chi_square <= critical_value
        
        self.results = {
            "test_name": "Chi-cuadrado de Bondad de Ajuste",
            "sample_size": sum(observed),
            "chi_square_statistic": round(chi_square, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "critical_value": round(critical_value, 4),
            "p_value": round(p_value, 6),
            "alpha": alpha,
            "fits_distribution": fits_distribution,
            "observed_frequencies": observed,
            "expected_frequencies": adjusted_expected,
            "decision": "No rechazar H0 (Se ajusta a la distribución)" if fits_distribution else "Rechazar H0 (No se ajusta)"
        }
        
        return self.results
    
    def test_poisson(self, data: List[int], lambda_param: float = None, 
                    alpha: float = 0.05) -> Dict:
        """
        Prueba de Chi-cuadrado para distribución Poisson
        
        Args:
            data: Datos enteros
            lambda_param: Parámetro lambda (si es None, se calcula de los datos)
            alpha: Nivel de significancia
        
        Returns:
            Dict con resultados de la prueba
        """
        if not data:
            return {"error": "No hay datos para probar"}
        
        # Calcular lambda si no se proporciona
        if lambda_param is None:
            lambda_param = np.mean(data)
        
        # Obtener valores únicos y sus frecuencias
        unique_values, observed_counts = np.unique(data, return_counts=True)
        
        # Calcular frecuencias esperadas
        expected_counts = []
        for value in unique_values:
            prob = stats.poisson.pmf(value, lambda_param)
            expected_counts.append(prob * len(data))
        
        # Agrupar categorías con frecuencias esperadas bajas
        observed_adj, expected_adj = self._group_categories(
            list(observed_counts), list(expected_counts), min_expected=5
        )
        
        return self.test_custom_distribution(observed_adj, expected_adj, alpha)
    
    def test_binomial(self, data: List[int], n: int, p: float, 
                     alpha: float = 0.05) -> Dict:
        """
        Prueba de Chi-cuadrado para distribución Binomial
        
        Args:
            data: Datos enteros
            n: Número de ensayos
            p: Probabilidad de éxito
            alpha: Nivel de significancia
        
        Returns:
            Dict con resultados de la prueba
        """
        if not data:
            return {"error": "No hay datos para probar"}
        
        # Obtener valores únicos y sus frecuencias
        unique_values, observed_counts = np.unique(data, return_counts=True)
        
        # Calcular frecuencias esperadas
        expected_counts = []
        for value in unique_values:
            if 0 <= value <= n:
                prob = stats.binom.pmf(value, n, p)
                expected_counts.append(prob * len(data))
            else:
                expected_counts.append(0)
        
        # Agrupar categorías con frecuencias esperadas bajas
        observed_adj, expected_adj = self._group_categories(
            list(observed_counts), list(expected_counts), min_expected=5
        )
        
        return self.test_custom_distribution(observed_adj, expected_adj, alpha)
    
    def test_normal(self, data: List[float], mu: float = None, 
                   sigma: float = None, intervals: int = 10, 
                   alpha: float = 0.05) -> Dict:
        """
        Prueba de Chi-cuadrado para distribución Normal
        
        Args:
            data: Datos continuos
            mu: Media (si es None, se calcula de los datos)
            sigma: Desviación estándar (si es None, se calcula de los datos)
            intervals: Número de intervalos
            alpha: Nivel de significancia
        
        Returns:
            Dict con resultados de la prueba
        """
        if not data:
            return {"error": "No hay datos para probar"}
        
        # Calcular parámetros si no se proporcionan
        if mu is None:
            mu = np.mean(data)
        if sigma is None:
            sigma = np.std(data)
        
        n = len(data)
        observed_frequencies = [0] * intervals
        
        # Crear intervalos basados en la distribución normal
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        
        # Usar percentiles de la distribución normal para los límites
        percentiles = np.linspace(0, 100, intervals + 1)
        boundaries = np.percentile(data, percentiles)
        
        # Contar frecuencias observadas
        for value in data:
            for i in range(intervals):
                if boundaries[i] <= value < boundaries[i + 1]:
                    observed_frequencies[i] += 1
                    break
            else:
                # Último intervalo incluye el valor máximo
                if value == boundaries[-1]:
                    observed_frequencies[-1] += 1
        
        # Calcular frecuencias esperadas
        expected_frequencies = []
        for i in range(intervals):
            prob = (stats.norm.cdf(boundaries[i + 1], mu, sigma) - 
                   stats.norm.cdf(boundaries[i], mu, sigma))
            expected_frequencies.append(prob * n)
        
        # Agrupar categorías con frecuencias esperadas bajas
        observed_adj, expected_adj = self._group_categories(
            observed_frequencies, expected_frequencies, min_expected=5
        )
        
        return self.test_custom_distribution(observed_adj, expected_adj, alpha)
    
    def _get_critical_value(self, df: int, alpha: float) -> float:
        """
        Obtiene el valor crítico de Chi-cuadrado
        
        Args:
            df: Grados de libertad
            alpha: Nivel de significancia
        
        Returns:
            Valor crítico
        """
        # Usar tabla predefinida o scipy si no está en la tabla
        if alpha in self.critical_values and df in self.critical_values[alpha]:
            return self.critical_values[alpha][df]
        else:
            return stats.chi2.ppf(1 - alpha, df)
    
    def _adjust_expected_frequencies(self, expected: List[float], 
                                   min_frequency: float = 5) -> List[float]:
        """
        Ajusta frecuencias esperadas para cumplir con los requisitos del test
        
        Args:
            expected: Frecuencias esperadas
            min_frequency: Frecuencia mínima permitida
        
        Returns:
            Frecuencias ajustadas
        """
        adjusted = expected.copy()
        
        # Combinar categorías adyacentes con frecuencias bajas
        i = 0
        while i < len(adjusted):
            if adjusted[i] < min_frequency:
                # Buscar categoría adyacente para combinar
                if i < len(adjusted) - 1:
                    adjusted[i] += adjusted[i + 1]
                    adjusted.pop(i + 1)
                elif i > 0:
                    adjusted[i - 1] += adjusted[i]
                    adjusted.pop(i)
                else:
                    # Única categoría, no se puede combinar
                    adjusted[i] = min_frequency
                    break
            else:
                i += 1
        
        return adjusted
    
    def _group_categories(self, observed: List[float], expected: List[float], 
                         min_expected: float = 5) -> Tuple[List[float], List[float]]:
        """
        Agrupa categorías con frecuencias esperadas bajas
        
        Args:
            observed: Frecuencias observadas
            expected: Frecuencias esperadas
            min_expected: Frecuencia mínima esperada
        
        Returns:
            Tupla con listas ajustadas (observed, expected)
        """
        observed_adj = []
        expected_adj = []
        
        i = 0
        while i < len(observed):
            current_obs = observed[i]
            current_exp = expected[i]
            
            # Si la frecuencia esperada es muy baja, agrupar con la siguiente
            if current_exp < min_expected and i < len(observed) - 1:
                j = i + 1
                while j < len(observed) and current_exp < min_expected:
                    current_obs += observed[j]
                    current_exp += expected[j]
                    j += 1
                i = j
            else:
                i += 1
            
            observed_adj.append(current_obs)
            expected_adj.append(current_exp)
        
        return observed_adj, expected_adj
    
    def get_detailed_report(self) -> str:
        """
        Genera un reporte detallado de la última prueba realizada
        """
        if not self.results:
            return "No se ha realizado ninguna prueba"
        
        report = f"=== {self.results['test_name']} ===\n\n"
        
        # Información básica
        report += f"Muestra: {self.results.get('sample_size', 'N/A')} observaciones\n"
        report += f"Estadístico Chi-cuadrado: {self.results['chi_square_statistic']}\n"
        report += f"Grados de libertad: {self.results['degrees_of_freedom']}\n"
        report += f"Valor crítico (α={self.results['alpha']}): {self.results['critical_value']}\n"
        report += f"Valor p: {self.results['p_value']}\n\n"
        
        # Decisión
        report += f"DECISIÓN: {self.results['decision']}\n\n"
        
        # Frecuencias
        if 'observed_frequencies' in self.results:
            report += "Frecuencias observadas:\n"
            report += f"{self.results['observed_frequencies']}\n\n"
        
        if 'expected_frequency' in self.results:
            report += f"Frecuencia esperada por intervalo: {self.results['expected_frequency']:.2f}\n"
        elif 'expected_frequencies' in self.results:
            report += "Frecuencias esperadas:\n"
            report += f"{[round(x, 2) for x in self.results['expected_frequencies']]}\n"
        
        return report
    
    def get_interpretation(self) -> str:
        """
        Proporciona una interpretación de los resultados
        """
        if not self.results:
            return "No hay resultados para interpretar"
        
        chi_square = self.results['chi_square_statistic']
        critical = self.results['critical_value']
        p_value = self.results['p_value']
        alpha = self.results['alpha']
        
        interpretation = f"Interpretación:\n"
        interpretation += f"El estadístico Chi-cuadrado calculado es {chi_square:.4f}\n"
        interpretation += f"El valor crítico para α={alpha} es {critical:.4f}\n"
        
        if chi_square <= critical:
            interpretation += f"Como {chi_square:.4f} ≤ {critical:.4f}, no rechazamos la hipótesis nula.\n"
            interpretation += "Esto sugiere que los datos siguen la distribución especificada.\n"
        else:
            interpretation += f"Como {chi_square:.4f} > {critical:.4f}, rechazamos la hipótesis nula.\n"
            interpretation += "Esto sugiere que los datos NO siguen la distribución especificada.\n"
        
        interpretation += f"\nEl valor p es {p_value:.6f}. "
        if p_value > alpha:
            interpretation += f"Como p > α, no hay evidencia suficiente para rechazar H0."
        else:
            interpretation += f"Como p ≤ α, hay evidencia suficiente para rechazar H0."
        
        return interpretation

# Funciones de conveniencia
def quick_uniformity_test(numbers: List[float], intervals: int = 10, 
                         alpha: float = 0.05) -> Dict:
    """
    Función rápida para prueba de uniformidad
    """
    tester = ChiSquareTest()
    return tester.test_uniformity(numbers, intervals, alpha)

def quick_goodness_of_fit(observed: List[float], expected: List[float], 
                         alpha: float = 0.05) -> Dict:
    """
    Función rápida para prueba de bondad de ajuste
    """
    tester = ChiSquareTest()
    return tester.test_custom_distribution(observed, expected, alpha)

if __name__ == "__main__":
    # Ejemplo de uso
    print("=== Ejemplo de Prueba Chi-cuadrado ===\n")
    
    # Generar datos de ejemplo
    np.random.seed(42)
    sample_data = np.random.uniform(0, 1, 1000)
    
    # Realizar prueba de uniformidad
    tester = ChiSquareTest()
    results = tester.test_uniformity(sample_data)
    
    print(tester.get_detailed_report())
    print(tester.get_interpretation())