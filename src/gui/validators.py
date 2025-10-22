"""
Módulo de validación de datos y parámetros para la aplicación de simulación estadística
Validaciones para generadores, distribuciones, autómatas y pruebas estadísticas
"""

import re
import numpy as np
from typing import Union, Tuple, List, Optional, Any
from tkinter import messagebox

class DataValidator:
    """
    Clase principal para validación de datos y parámetros
    """
    
    # Patrones regex para validación
    PATTERNS = {
        'integer': r'^-?\d+$',
        'float': r'^-?\d*\.?\d+([eE][-+]?\d+)?$',
        'positive_integer': r'^\d+$',
        'positive_float': r'^\d*\.?\d+([eE][-+]?\d+)?$',
        'probability': r'^(0(\.\d+)?|1(\.0+)?)$',
        'percentage': r'^(100(\.0+)?|\d{1,2}(\.\d+)?)$',
        'scientific': r'^-?\d*\.?\d+[eE][-+]?\d+$'
    }
    
    # Límites y rangos por defecto
    DEFAULT_LIMITS = {
        'probability': (0.0, 1.0),
        'percentage': (0.0, 100.0),
        'positive_int': (1, 10**9),
        'positive_float': (0.0, 10**9),
        'rule_number': (0, 255),
        'sample_size': (1, 1000000),
        'grid_size': (1, 1000),
        'lambda_param': (0.001, 1000.0),
        'sigma_param': (0.001, 1000.0)
    }
    
    @staticmethod
    def validate_number(value: str, 
                       number_type: str = 'float',
                       min_val: Optional[float] = None,
                       max_val: Optional[float] = None,
                       allow_empty: bool = False) -> Tuple[bool, Union[float, int, None]]:
        """
        Valida un valor numérico
        
        Args:
            value: Valor a validar
            number_type: Tipo de número ('integer', 'float', 'positive_integer', etc.)
            min_val: Valor mínimo permitido
            max_val: Valor máximo permitido
            allow_empty: Si permite valores vacíos
        
        Returns:
            Tuple (éxito, valor convertido o mensaje de error)
        """
        # Validar vacío
        if not value.strip():
            if allow_empty:
                return True, None
            else:
                return False, "Este campo no puede estar vacío"
        
        # Validar patrón
        pattern = DataValidator.PATTERNS.get(number_type, DataValidator.PATTERNS['float'])
        if not re.match(pattern, value.strip()):
            return False, f"Formato inválido para {number_type}"
        
        try:
            # Convertir al tipo apropiado
            if number_type in ['integer', 'positive_integer']:
                converted = int(value)
            else:
                converted = float(value)
            
            # Validar rango
            if min_val is not None and converted < min_val:
                return False, f"El valor debe ser mayor o igual a {min_val}"
            
            if max_val is not None and converted > max_val:
                return False, f"El valor debe ser menor o igual a {max_val}"
            
            # Validaciones específicas por tipo
            if number_type == 'positive_integer' and converted <= 0:
                return False, "El valor debe ser un entero positivo"
            
            if number_type == 'probability' and not (0 <= converted <= 1):
                return False, "La probabilidad debe estar entre 0 y 1"
            
            if number_type == 'percentage' and not (0 <= converted <= 100):
                return False, "El porcentaje debe estar entre 0 y 100"
            
            return True, converted
            
        except (ValueError, TypeError) as e:
            return False, f"Error de conversión: {str(e)}"
    
    @staticmethod
    def validate_list(values: str, 
                     separator: str = ',',
                     expected_type: str = 'float',
                     min_count: int = 1,
                     max_count: Optional[int] = None) -> Tuple[bool, Union[List, str]]:
        """
        Valida una lista de valores separados
        
        Args:
            values: String con valores separados
            separator: Carácter separador
            expected_type: Tipo de valores esperados
            min_count: Mínimo número de elementos
            max_count: Máximo número de elementos
        
        Returns:
            Tuple (éxito, lista convertida o mensaje de error)
        """
        if not values.strip():
            return False, "La lista no puede estar vacía"
        
        items = [item.strip() for item in values.split(separator) if item.strip()]
        
        # Validar cantidad
        if len(items) < min_count:
            return False, f"Se requieren al menos {min_count} elementos"
        
        if max_count and len(items) > max_count:
            return False, f"No se permiten más de {max_count} elementos"
        
        # Validar cada elemento
        converted_list = []
        for i, item in enumerate(items, 1):
            success, result = DataValidator.validate_number(item, expected_type)
            if not success:
                return False, f"Elemento {i} inválido: {result}"
            converted_list.append(result)
        
        return True, converted_list

class GeneratorValidator:
    """
    Validaciones específicas para generadores de números aleatorios
    """
    
    @staticmethod
    def validate_lcg_parameters(seed: str, a: str, c: str, m: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para generador congruencial lineal
        """
        params = {}
        validations = [
            ('seed', seed, 'positive_integer', (1, 2**32)),
            ('a', a, 'positive_integer', (1, 2**32)),
            ('c', c, 'positive_integer', (0, 2**32)),
            ('m', m, 'positive_integer', (1, 2**32))
        ]
        
        for param_name, value, value_type, limits in validations:
            success, result = DataValidator.validate_number(value, value_type, *limits)
            if not success:
                return False, f"Parámetro {param_name}: {result}"
            params[param_name] = result
        
        # Validaciones específicas LCG
        if params['m'] <= 0:
            return False, "El módulo (m) debe ser positivo"
        
        if params['a'] >= params['m']:
            return False, "El multiplicador (a) debe ser menor que el módulo (m)"
        
        if params['c'] >= params['m']:
            return False, "El incremento (c) debe ser menor que el módulo (m)"
        
        return True, params
    
    @staticmethod
    def validate_middle_square_parameters(seed: str, digits: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para generador de cuadrados medios
        """
        params = {}
        
        # Validar semilla
        success, result = DataValidator.validate_number(seed, 'positive_integer', (1, 10**9))
        if not success:
            return False, f"Semilla: {result}"
        params['seed'] = result
        
        # Validar dígitos
        success, result = DataValidator.validate_number(digits, 'positive_integer', (2, 10))
        if not success:
            return False, f"Dígitos: {result}"
        params['digits'] = result
        
        # Validar que la semilla tenga los dígitos correctos
        seed_str = str(params['seed'])
        if len(seed_str) != params['digits']:
            return False, f"La semilla debe tener exactamente {params['digits']} dígitos"
        
        return True, params
    
    @staticmethod
    def validate_fibonacci_parameters(seed1: str, seed2: str, m: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para generador de Fibonacci
        """
        params = {}
        validations = [
            ('seed1', seed1, 'positive_integer', (1, 2**32)),
            ('seed2', seed2, 'positive_integer', (1, 2**32)),
            ('m', m, 'positive_integer', (1, 2**32))
        ]
        
        for param_name, value, value_type, limits in validations:
            success, result = DataValidator.validate_number(value, value_type, *limits)
            if not success:
                return False, f"Parámetro {param_name}: {result}"
            params[param_name] = result
        
        # Validar que las semillas sean diferentes
        if params['seed1'] == params['seed2']:
            return False, "Las semillas deben ser diferentes"
        
        return True, params
    
    @staticmethod
    def validate_sample_size(size: str, max_size: int = 100000) -> Tuple[bool, int]:
        """
        Valida el tamaño de muestra
        """
        success, result = DataValidator.validate_number(
            size, 'positive_integer', (1, max_size)
        )
        if not success:
            return False, f"Tamaño de muestra: {result}"
        return True, result

class DistributionValidator:
    """
    Validaciones específicas para distribuciones de probabilidad
    """
    
    @staticmethod
    def validate_uniform_parameters(a: str, b: str, continuous: bool = True) -> Tuple[bool, dict]:
        """
        Valida parámetros para distribución uniforme
        """
        params = {}
        
        if continuous:
            validations = [
                ('a', a, 'float', None, None),
                ('b', b, 'float', None, None)
            ]
        else:
            validations = [
                ('a', a, 'integer', None, None),
                ('b', b, 'integer', None, None)
            ]
        
        for param_name, value, value_type, min_val, max_val in validations:
            success, result = DataValidator.validate_number(value, value_type, min_val, max_val)
            if not success:
                return False, f"Parámetro {param_name}: {result}"
            params[param_name] = result
        
        # Validar que a < b
        if params['a'] >= params['b']:
            return False, "El parámetro 'a' debe ser menor que 'b'"
        
        return True, params
    
    @staticmethod
    def validate_erlang_parameters(k: str, lambd: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para distribución K-Erlang
        """
        params = {}
        
        # Validar k (debe ser entero positivo)
        success, result = DataValidator.validate_number(k, 'positive_integer', (1, 100))
        if not success:
            return False, f"Parámetro k: {result}"
        params['k'] = result
        
        # Validar lambda (debe ser positivo)
        success, result = DataValidator.validate_number(lambd, 'positive_float', (0.001, 1000))
        if not success:
            return False, f"Parámetro λ: {result}"
        params['lambda'] = result
        
        return True, params
    
    @staticmethod
    def validate_exponential_parameters(lambd: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para distribución exponencial
        """
        params = {}
        
        success, result = DataValidator.validate_number(lambd, 'positive_float', (0.001, 1000))
        if not success:
            return False, f"Parámetro λ: {result}"
        params['lambda'] = result
        
        return True, params
    
    @staticmethod
    def validate_normal_parameters(mu: str, sigma: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para distribución normal
        """
        params = {}
        
        validations = [
            ('mu', mu, 'float', None, None),
            ('sigma', sigma, 'positive_float', (0.001, 1000), None)
        ]
        
        for param_name, value, value_type, min_val, max_val in validations:
            success, result = DataValidator.validate_number(value, value_type, min_val, max_val)
            if not success:
                return False, f"Parámetro {param_name}: {result}"
            params[param_name] = result
        
        return True, params
    
    @staticmethod
    def validate_binomial_parameters(n: str, p: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para distribución binomial
        """
        params = {}
        
        # Validar n (entero positivo)
        success, result = DataValidator.validate_number(n, 'positive_integer', (1, 1000))
        if not success:
            return False, f"Parámetro n: {result}"
        params['n'] = result
        
        # Validar p (probabilidad)
        success, result = DataValidator.validate_number(p, 'probability')
        if not success:
            return False, f"Parámetro p: {result}"
        params['p'] = result
        
        return True, params
    
    @staticmethod
    def validate_poisson_parameters(lambd: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para distribución Poisson
        """
        params = {}
        
        success, result = DataValidator.validate_number(lambd, 'positive_float', (0.001, 1000))
        if not success:
            return False, f"Parámetro λ: {result}"
        params['lambda'] = result
        
        return True, params

class AutomataValidator:
    """
    Validaciones específicas para autómatas celulares
    """
    
    @staticmethod
    def validate_grid_size(rows: str, cols: str, max_size: int = 500) -> Tuple[bool, dict]:
        """
        Valida dimensiones de la cuadrícula
        """
        params = {}
        validations = [
            ('rows', rows, 'positive_integer', (1, max_size)),
            ('cols', cols, 'positive_integer', (1, max_size))
        ]
        
        for param_name, value, value_type, limits in validations:
            success, result = DataValidator.validate_number(value, value_type, *limits)
            if not success:
                return False, f"{param_name.capitalize()}: {result}"
            params[param_name] = result
        
        # Validar límites de rendimiento
        total_cells = params['rows'] * params['cols']
        if total_cells > 10000:
            return False, f"La cuadrícula es muy grande ({total_cells} celdas). Máximo 10000 celdas."
        
        return True, params
    
    @staticmethod
    def validate_rule_number(rule: str) -> Tuple[bool, int]:
        """
        Valida número de regla para autómatas unidimensionales
        """
        success, result = DataValidator.validate_number(rule, 'integer', (0, 255))
        if not success:
            return False, f"Regla: {result}"
        return True, result
    
    @staticmethod
    def validate_density(density: str) -> Tuple[bool, float]:
        """
        Valida densidad para inicialización aleatoria
        """
        success, result = DataValidator.validate_number(density, 'probability')
        if not success:
            return False, f"Densidad: {result}"
        return True, result
    
    @staticmethod
    def validate_covid_parameters(infection_rate: str, recovery_rate: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para simulación COVID
        """
        params = {}
        validations = [
            ('infection_rate', infection_rate, 'probability'),
            ('recovery_rate', recovery_rate, 'probability')
        ]
        
        for param_name, value, value_type in validations:
            success, result = DataValidator.validate_number(value, value_type)
            if not success:
                return False, f"Tasa de {param_name}: {result}"
            params[param_name] = result
        
        return True, params

class StatisticalTestValidator:
    """
    Validaciones para pruebas estadísticas
    """
    
    @staticmethod
    def validate_chi_square_parameters(intervals: str, alpha: str) -> Tuple[bool, dict]:
        """
        Valida parámetros para prueba Chi-cuadrado
        """
        params = {}
        
        # Validar número de intervalos
        success, result = DataValidator.validate_number(intervals, 'positive_integer', (2, 100))
        if not success:
            return False, f"Intervalos: {result}"
        params['intervals'] = result
        
        # Validar nivel de significancia
        success, result = DataValidator.validate_number(alpha, 'probability', (0.001, 0.2))
        if not success:
            return False, f"Alpha: {result}"
        params['alpha'] = result
        
        return True, params
    
    @staticmethod
    def validate_data_for_tests(data: List, min_sample_size: int = 10) -> Tuple[bool, str]:
        """
        Valida datos para pruebas estadísticas
        """
        if not data:
            return False, "No hay datos para analizar"
        
        if len(data) < min_sample_size:
            return False, f"Se requieren al menos {min_sample_size} observaciones"
        
        # Verificar que todos los elementos sean numéricos
        for i, value in enumerate(data):
            if not isinstance(value, (int, float)):
                return False, f"Elemento {i} no es numérico: {value}"
        
        # Verificar rango para pruebas de uniformidad
        if any(x < 0 or x > 1 for x in data):
            return False, "Los datos deben estar en el rango [0, 1] para pruebas de uniformidad"
        
        return True, "Datos válidos"

class InputHandler:
    """
    Manejador de entrada con validación y mensajes de error
    """
    
    @staticmethod
    def validate_and_show_error(parent, validation_func, *args, **kwargs):
        """
        Ejecuta validación y muestra mensaje de error si falla
        """
        success, result = validation_func(*args, **kwargs)
        
        if not success:
            messagebox.showerror("Error de Validación", result, parent=parent)
            return False, None
        else:
            return True, result
    
    @staticmethod
    def validate_multiple_fields(parent, validations):
        """
        Valida múltiples campos y muestra errores acumulados
        """
        errors = []
        results = {}
        
        for field_name, validation_func, validation_args in validations:
            success, result = validation_func(*validation_args)
            if not success:
                errors.append(f"{field_name}: {result}")
            else:
                results[field_name] = result
        
        if errors:
            error_message = "Errores de validación:\n\n" + "\n".join(errors)
            messagebox.showerror("Errores de Validación", error_message, parent=parent)
            return False, None
        else:
            return True, results

# Funciones de conveniencia para validación rápida
def validate_positive_integer(value: str, field_name: str = "Valor") -> Tuple[bool, Union[int, str]]:
    """Valida un entero positivo rápidamente"""
    return DataValidator.validate_number(value, 'positive_integer', 1, 100000)

def validate_probability(value: str, field_name: str = "Probabilidad") -> Tuple[bool, Union[float, str]]:
    """Valida una probabilidad rápidamente"""
    return DataValidator.validate_number(value, 'probability')

def validate_percentage(value: str, field_name: str = "Porcentaje") -> Tuple[bool, Union[float, str]]:
    """Valida un porcentaje rápidamente"""
    return DataValidator.validate_number(value, 'percentage')

def validate_float_range(value: str, min_val: float, max_val: float, field_name: str = "Valor") -> Tuple[bool, Union[float, str]]:
    """Valida un float en un rango específico"""
    return DataValidator.validate_number(value, 'float', min_val, max_val)

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplos de validación
    print("=== Ejemplos de Validación ===\n")
    
    # Validar número positivo
    success, result = validate_positive_integer("42")
    print(f"Validar 42: {success}, {result}")
    
    # Validar probabilidad
    success, result = validate_probability("0.75")
    print(f"Validar 0.75: {success}, {result}")
    
    # Validar parámetros LCG
    success, result = GeneratorValidator.validate_lcg_parameters("12345", "1664525", "1013904223", "4294967296")
    print(f"Validar LCG: {success}, {result}")
    
    # Validar distribución normal
    success, result = DistributionValidator.validate_normal_parameters("0", "1")
    print(f"Validar Normal(0,1): {success}, {result}")