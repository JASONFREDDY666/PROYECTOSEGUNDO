"""
Módulo completo de distribuciones de probabilidad
Implementa distribuciones continuas y discretas para generación de variables aleatorias
"""

import numpy as np
import math
import scipy.special as special
from typing import List, Union, Dict, Optional
import random

class ContinuousDistributions:
    """
    Distribuciones de probabilidad continuas
    """
    
    @staticmethod
    def uniform(a: float, b: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Uniforme Continua U(a, b)
        
        Args:
            a: Límite inferior
            b: Límite superior  
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos uniformemente
        """
        if a >= b:
            raise ValueError("El parámetro 'a' debe ser menor que 'b'")
        
        return [round(a + (b - a) * random.random(), decimals) for _ in range(size)]
    
    @staticmethod
    def erlang(k: int, lambd: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución K-Erlang (Gamma con shape entero)
        
        Args:
            k: Parámetro de shape (entero positivo)
            lambd: Parámetro de tasa (positivo)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos según K-Erlang
        """
        if k <= 0:
            raise ValueError("El parámetro 'k' debe ser un entero positivo")
        if lambd <= 0:
            raise ValueError("El parámetro 'lambda' debe ser positivo")
        
        # K-Erlang es la suma de k variables exponenciales independientes
        results = []
        for _ in range(size):
            # Sumar k exponenciales
            total = sum(-math.log(1 - random.random()) / lambd for _ in range(k))
            results.append(round(total, decimals))
        
        return results
    
    @staticmethod
    def exponential(lambd: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Exponencial
        
        Args:
            lambd: Parámetro de tasa (positivo)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos exponencialmente
        """
        if lambd <= 0:
            raise ValueError("El parámetro 'lambda' debe ser positivo")
        
        # Método de transformada inversa
        return [round(-math.log(1 - random.random()) / lambd, decimals) for _ in range(size)]
    
    @staticmethod
    def normal(mu: float, sigma: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Normal (Box-Muller)
        
        Args:
            mu: Media
            sigma: Desviación estándar (positiva)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos normalmente
        """
        if sigma <= 0:
            raise ValueError("La desviación estándar debe ser positiva")
        
        results = []
        # Box-Muller transform
        for i in range(size):
            if i % 2 == 0:
                u1 = random.random()
                u2 = random.random()
                z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
                results.append(round(mu + sigma * z0, decimals))
                if len(results) < size:
                    results.append(round(mu + sigma * z1, decimals))
        
        return results[:size]
    
    @staticmethod
    def gamma(alpha: float, beta: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Gamma
        
        Args:
            alpha: Parámetro de shape (positivo)
            beta: Parámetro de scale (positivo)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos según Gamma
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Los parámetros alpha y beta deben ser positivos")
        
        results = []
        for _ in range(size):
            # Método de aceptación-rechazo para Gamma general
            if alpha < 1:
                # Usar método de transformada inversa modificado
                b = (math.e + alpha) / math.e
                while True:
                    u1 = random.random()
                    u2 = random.random()
                    p = b * u1
                    if p <= 1:
                        x = p ** (1 / alpha)
                        if u2 <= math.exp(-x):
                            results.append(round(x * beta, decimals))
                            break
                    else:
                        x = -math.log((b - p) / alpha)
                        if u2 <= x ** (alpha - 1):
                            results.append(round(x * beta, decimals))
                            break
            else:
                # Método de Cheng para alpha >= 1
                a = alpha
                b = a - 1
                c = 3 * a - 0.75
                while True:
                    u1 = random.random()
                    u2 = random.random()
                    w = u1 * (1 - u1)
                    y = math.sqrt(c / w) * (u1 - 0.5)
                    x = b + y
                    if x >= 0:
                        z = 64 * (w ** 3) * (u2 ** 2)
                        if z <= 1 - (2 * y ** 2) / x or math.log(z) <= 2 * (b * math.log(x / b) - y):
                            results.append(round(x * beta, decimals))
                            break
        
        return results
    
    @staticmethod
    def weibull(alpha: float, beta: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Weibull
        
        Args:
            alpha: Parámetro de shape (positivo)
            beta: Parámetro de scale (positivo)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos según Weibull
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Los parámetros alpha y beta deben ser positivos")
        
        # Método de transformada inversa
        return [round(beta * (-math.log(1 - random.random())) ** (1/alpha), decimals) for _ in range(size)]
    
    @staticmethod
    def beta(alpha: float, beta: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Beta
        
        Args:
            alpha: Parámetro alpha (positivo)
            beta: Parámetro beta (positivo)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos según Beta
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Los parámetros alpha y beta deben ser positivos")
        
        results = []
        for _ in range(size):
            # Generar usando relación con distribuciones Gamma
            x1 = random.gammavariate(alpha, 1)
            x2 = random.gammavariate(beta, 1)
            results.append(round(x1 / (x1 + x2), decimals))
        
        return results
    
    @staticmethod
    def lognormal(mu: float, sigma: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Log-Normal
        
        Args:
            mu: Media del logaritmo
            sigma: Desviación estándar del logaritmo (positiva)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos según Log-Normal
        """
        if sigma <= 0:
            raise ValueError("La desviación estándar debe ser positiva")
        
        # Generar normal y aplicar exponencial
        normals = ContinuousDistributions.normal(mu, sigma, size)
        return [round(math.exp(x), decimals) for x in normals]
    
    @staticmethod
    def triangular(low: float, high: float, mode: float, size: int, decimals: int = 4) -> List[float]:
        """
        Distribución Triangular
        
        Args:
            low: Límite inferior
            high: Límite superior
            mode: Moda (entre low y high)
            size: Tamaño de la muestra
            decimals: Decimales para redondeo
        
        Returns:
            Lista de números distribuidos triangularmente
        """
        if not (low <= mode <= high):
            raise ValueError("Debe cumplirse: low <= mode <= high")
        
        results = []
        for _ in range(size):
            u = random.random()
            if u <= (mode - low) / (high - low):
                x = low + math.sqrt(u * (high - low) * (mode - low))
            else:
                x = high - math.sqrt((1 - u) * (high - low) * (high - mode))
            results.append(round(x, decimals))
        
        return results

class DiscreteDistributions:
    """
    Distribuciones de probabilidad discretas
    """
    
    @staticmethod
    def uniform(a: int, b: int, size: int) -> List[int]:
        """
        Distribución Uniforme Discreta
        
        Args:
            a: Límite inferior (entero)
            b: Límite superior (entero)
            size: Tamaño de la muestra
        
        Returns:
            Lista de enteros distribuidos uniformemente
        """
        if a >= b:
            raise ValueError("El parámetro 'a' debe ser menor que 'b'")
        
        return [random.randint(a, b) for _ in range(size)]
    
    @staticmethod
    def bernoulli(p: float, size: int) -> List[int]:
        """
        Distribución Bernoulli
        
        Args:
            p: Probabilidad de éxito (0 <= p <= 1)
            size: Tamaño de la muestra
        
        Returns:
            Lista de 0s y 1s distribuidos según Bernoulli
        """
        if not 0 <= p <= 1:
            raise ValueError("La probabilidad p debe estar entre 0 y 1")
        
        return [1 if random.random() < p else 0 for _ in range(size)]
    
    @staticmethod
    def binomial(n: int, p: float, size: int) -> List[int]:
        """
        Distribución Binomial
        
        Args:
            n: Número de ensayos (entero positivo)
            p: Probabilidad de éxito (0 <= p <= 1)
            size: Tamaño de la muestra
        
        Returns:
            Lista de enteros distribuidos binomialmente
        """
        if n <= 0:
            raise ValueError("El número de ensayos n debe ser positivo")
        if not 0 <= p <= 1:
            raise ValueError("La probabilidad p debe estar entre 0 y 1")
        
        # Método directo: suma de n Bernoullis
        return [sum(1 if random.random() < p else 0 for _ in range(n)) for _ in range(size)]
    
    @staticmethod
    def poisson(lambd: float, size: int) -> List[int]:
        """
        Distribución Poisson
        
        Args:
            lambd: Parámetro de tasa (positivo)
            size: Tamaño de la muestra
        
        Returns:
            Lista de enteros distribuidos según Poisson
        """
        if lambd <= 0:
            raise ValueError("El parámetro lambda debe ser positivo")
        
        results = []
        for _ in range(size):
            # Algoritmo de Knuth
            L = math.exp(-lambd)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= random.random()
            results.append(k - 1)
        
        return results
    
    @staticmethod
    def geometric(p: float, size: int) -> List[int]:
        """
        Distribución Geométrica
        
        Args:
            p: Probabilidad de éxito (0 < p <= 1)
            size: Tamaño de la muestra
        
        Returns:
            Lista de enteros distribuidos geométricamente
        """
        if not 0 < p <= 1:
            raise ValueError("La probabilidad p debe estar entre 0 y 1")
        
        # Método de transformada inversa
        return [math.floor(math.log(random.random()) / math.log(1 - p)) for _ in range(size)]
    
    @staticmethod
    def negative_binomial(r: int, p: float, size: int) -> List[int]:
        """
        Distribución Binomial Negativa
        
        Args:
            r: Número de éxitos requeridos (entero positivo)
            p: Probabilidad de éxito (0 < p <= 1)
            size: Tamaño de la muestra
        
        Returns:
            Lista de enteros distribuidos según binomial negativa
        """
        if r <= 0:
            raise ValueError("El parámetro r debe ser positivo")
        if not 0 < p <= 1:
            raise ValueError("La probabilidad p debe estar entre 0 y 1")
        
        # Suma de r geométricas
        results = []
        for _ in range(size):
            total = sum(DiscreteDistributions.geometric(p, 1)[0] for _ in range(r))
            results.append(total)
        
        return results
    
    @staticmethod
    def hypergeometric(N: int, K: int, n: int, size: int) -> List[int]:
        """
        Distribución Hipergeométrica
        
        Args:
            N: Tamaño de la población
            K: Número de éxitos en la población
            n: Tamaño de la muestra
            size: Número de muestras
        
        Returns:
            Lista de enteros distribuidos hipergeométricamente
        """
        if not (0 <= K <= N and 0 <= n <= N):
            raise ValueError("Parámetros inválidos: debe cumplirse 0 <= K <= N y 0 <= n <= N")
        
        results = []
        for _ in range(size):
            # Simulación directa del proceso de muestreo
            population = [1] * K + [0] * (N - K)
            random.shuffle(population)
            sample = random.sample(population, n)
            results.append(sum(sample))
        
        return results

class DistributionAnalyzer:
    """
    Analizador de distribuciones - Calcula estadísticas y propiedades
    """
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """
        Calcula estadísticas descriptivas de un conjunto de datos
        
        Args:
            data: Lista de valores numéricos
        
        Returns:
            Diccionario con estadísticas
        """
        if not data:
            return {}
        
        data_array = np.array(data)
        
        return {
            'count': len(data),
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'mode': float(DistributionAnalyzer._calculate_mode(data)),
            'std_dev': float(np.std(data_array)),
            'variance': float(np.var(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'range': float(np.max(data_array) - np.min(data_array)),
            'q1': float(np.percentile(data_array, 25)),
            'q3': float(np.percentile(data_array, 75)),
            'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
            'skewness': float(DistributionAnalyzer._calculate_skewness(data_array)),
            'kurtosis': float(DistributionAnalyzer._calculate_kurtosis(data_array))
        }
    
    @staticmethod
    def _calculate_mode(data: List[float]) -> float:
        """Calcula la moda de los datos"""
        if not data:
            return 0.0
        
        values, counts = np.unique(data, return_counts=True)
        max_count_index = np.argmax(counts)
        return float(values[max_count_index])
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calcula el coeficiente de asimetría"""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        skew = (np.sum((data - mean) ** 3) / n) / (std ** 3)
        return skew
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calcula la curtosis"""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        kurt = (np.sum((data - mean) ** 4) / n) / (std ** 4) - 3
        return kurt
    
    @staticmethod
    def calculate_empirical_cdf(data: List[float], x_values: List[float]) -> List[float]:
        """
        Calcula la función de distribución empírica
        
        Args:
            data: Datos de la muestra
            x_values: Valores de x para evaluar la CDF
        
        Returns:
            Valores de la CDF empírica en los puntos x_values
        """
        if not data:
            return [0.0] * len(x_values)
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        cdf_values = []
        
        for x in x_values:
            count = sum(1 for value in sorted_data if value <= x)
            cdf_values.append(count / n)
        
        return cdf_values
    
    @staticmethod
    def calculate_histogram(data: List[float], bins: int = 10) -> Dict:
        """
        Calcula histograma de los datos
        
        Args:
            data: Datos de la muestra
            bins: Número de intervalos
        
        Returns:
            Diccionario con información del histograma
        """
        if not data:
            return {'frequencies': [], 'bin_edges': []}
        
        frequencies, bin_edges = np.histogram(data, bins=bins)
        
        return {
            'frequencies': frequencies.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)],
            'relative_frequencies': (frequencies / len(data)).tolist()
        }

class DistributionFactory:
    """
    Fábrica de distribuciones - Interfaz unificada para generar distribuciones
    """
    
    @staticmethod
    def create_distribution(dist_type: str, parameters: Dict, size: int) -> List:
        """
        Crea una distribución basada en el tipo y parámetros
        
        Args:
            dist_type: Tipo de distribución
            parameters: Parámetros de la distribución
            size: Tamaño de la muestra
        
        Returns:
            Lista de valores generados
        """
        dist_type = dist_type.lower()
        
        try:
            if dist_type in ['uniform', 'uniforme']:
                return ContinuousDistributions.uniform(
                    parameters.get('a', 0), 
                    parameters.get('b', 1), 
                    size
                )
            
            elif dist_type in ['erlang', 'k-erlang']:
                return ContinuousDistributions.erlang(
                    parameters.get('k', 2),
                    parameters.get('lambda', 1.0),
                    size
                )
            
            elif dist_type in ['exponential', 'exponencial']:
                return ContinuousDistributions.exponential(
                    parameters.get('lambda', 1.0),
                    size
                )
            
            elif dist_type in ['normal', 'gaussian']:
                return ContinuousDistributions.normal(
                    parameters.get('mu', 0),
                    parameters.get('sigma', 1),
                    size
                )
            
            elif dist_type == 'gamma':
                return ContinuousDistributions.gamma(
                    parameters.get('alpha', 2),
                    parameters.get('beta', 2),
                    size
                )
            
            elif dist_type == 'weibull':
                return ContinuousDistributions.weibull(
                    parameters.get('alpha', 1),
                    parameters.get('beta', 1),
                    size
                )
            
            elif dist_type == 'beta':
                return ContinuousDistributions.beta(
                    parameters.get('alpha', 2),
                    parameters.get('beta', 2),
                    size
                )
            
            elif dist_type in ['uniform_d', 'uniforme_discreta']:
                return DiscreteDistributions.uniform(
                    parameters.get('a', 0),
                    parameters.get('b', 10),
                    size
                )
            
            elif dist_type == 'bernoulli':
                return DiscreteDistributions.bernoulli(
                    parameters.get('p', 0.5),
                    size
                )
            
            elif dist_type == 'binomial':
                return DiscreteDistributions.binomial(
                    parameters.get('n', 10),
                    parameters.get('p', 0.5),
                    size
                )
            
            elif dist_type == 'poisson':
                return DiscreteDistributions.poisson(
                    parameters.get('lambda', 3.0),
                    size
                )
            
            elif dist_type == 'geometric':
                return DiscreteDistributions.geometric(
                    parameters.get('p', 0.5),
                    size
                )
            
            else:
                raise ValueError(f"Tipo de distribución no soportado: {dist_type}")
                
        except Exception as e:
            raise ValueError(f"Error al crear distribución {dist_type}: {str(e)}")
    
    @staticmethod
    def get_distribution_info(dist_type: str) -> Dict:
        """
        Obtiene información sobre una distribución
        
        Args:
            dist_type: Tipo de distribución
        
        Returns:
            Información de la distribución
        """
        info = {
            'uniform': {
                'name': 'Uniforme Continua',
                'parameters': ['a (límite inferior)', 'b (límite superior)'],
                'description': 'Distribución donde todos los valores en el intervalo [a,b] son igualmente probables'
            },
            'erlang': {
                'name': 'K-Erlang',
                'parameters': ['k (shape entero)', 'λ (tasa)'],
                'description': 'Distribución de la suma de k variables exponenciales independientes'
            },
            'exponential': {
                'name': 'Exponencial', 
                'parameters': ['λ (tasa)'],
                'description': 'Distribución que modela el tiempo entre eventos en un proceso de Poisson'
            },
            'normal': {
                'name': 'Normal',
                'parameters': ['μ (media)', 'σ (desviación estándar)'],
                'description': 'Distribución en forma de campana, fundamental en estadística'
            },
            'binomial': {
                'name': 'Binomial',
                'parameters': ['n (ensayos)', 'p (probabilidad de éxito)'],
                'description': 'Distribución del número de éxitos en n ensayos independientes'
            },
            'poisson': {
                'name': 'Poisson',
                'parameters': ['λ (tasa)'],
                'description': 'Distribución que modela el número de eventos en un intervalo de tiempo'
            }
        }
        
        return info.get(dist_type.lower(), {'name': dist_type, 'parameters': [], 'description': 'Información no disponible'})

# Ejemplo de uso
if __name__ == "__main__":
    print("=== Ejemplos de Distribuciones ===\n")
    
    # Generar distribución K-Erlang
    erlang_data = ContinuousDistributions.erlang(k=2, lambd=1.0, size=1000)
    stats = DistributionAnalyzer.calculate_statistics(erlang_data)
    
    print("K-Erlang (k=2, λ=1.0):")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nMuestra: {erlang_data[:10]}...")
    
    # Usar fábrica de distribuciones
    binomial_data = DistributionFactory.create_distribution(
        'binomial', 
        {'n': 10, 'p': 0.5}, 
        1000
    )
    
    binomial_stats = DistributionAnalyzer.calculate_statistics(binomial_data)
    print(f"\nBinomial (n=10, p=0.5) - Media: {binomial_stats['mean']:.2f}")