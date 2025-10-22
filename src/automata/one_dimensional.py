"""
Autómatas celulares unidimensionales - COMPLETO
"""

import numpy as np
from typing import List, Dict
import random

class OneDimensionalAutomata:
    """Autómatas celulares unidimensionales con múltiples reglas"""
    
    def __init__(self, width: int, rule: int = 30):
        self.width = width
        self.rule = rule
        self.generation = 0
        self.history = []
        
        # Inicializar primera generación (una célula viva en el centro)
        self.current_state = np.zeros(width, dtype=int)
        if width > 0:
            self.current_state[width // 2] = 1
        self.history.append(self.current_state.copy())
        
    def set_rule(self, rule: int):
        """Establecer la regla del autómata (0-255)"""
        if 0 <= rule <= 255:
            self.rule = rule
    
    def get_rule_binary(self) -> str:
        """Obtener la regla en formato binario de 8 bits"""
        return format(self.rule, '08b')
    
    def get_rule_table(self) -> Dict[str, int]:
        """Obtener tabla de reglas para los 8 patrones posibles"""
        rule_binary = self.get_rule_binary()
        patterns = [
            '111', '110', '101', '100',
            '011', '010', '001', '000'
        ]
        return {pattern: int(bit) for pattern, bit in zip(patterns, rule_binary)}
    
    def apply_rule(self, left: int, center: int, right: int) -> int:
        """Aplicar la regla a un patrón de 3 células"""
        pattern = f"{left}{center}{right}"
        rule_table = self.get_rule_table()
        return rule_table.get(pattern, 0)
    
    def next_generation(self):
        """Calcular la siguiente generación"""
        if self.width == 0:
            return
            
        new_state = np.zeros(self.width, dtype=int)
        
        # Aplicar regla a cada célula (condiciones de contorno periódicas)
        for i in range(self.width):
            left = self.current_state[(i - 1) % self.width]
            center = self.current_state[i]
            right = self.current_state[(i + 1) % self.width]
            
            new_state[i] = self.apply_rule(left, center, right)
        
        self.current_state = new_state
        self.generation += 1
        self.history.append(new_state.copy())
    
    def random_initialization(self, density: float = 0.5):
        """Inicialización aleatoria"""
        self.current_state = np.random.choice([0, 1], size=self.width, 
                                            p=[1-density, density])
        self.generation = 0
        self.history = [self.current_state.copy()]
    
    def set_custom_initial_state(self, initial_state: List[int]):
        """Establecer un estado inicial personalizado"""
        if len(initial_state) == self.width:
            self.current_state = np.array(initial_state, dtype=int)
            self.generation = 0
            self.history = [self.current_state.copy()]
    
    def get_history_matrix(self, generations: int = None) -> np.ndarray:
        """Obtener matriz de historia para visualización"""
        if generations is None:
            generations = len(self.history)
        
        if len(self.history) < generations:
            # Simular las generaciones faltantes
            while len(self.history) < generations:
                self.next_generation()
        
        return np.array(self.history[:generations])
    
    def get_statistics(self) -> Dict:
        """Obtener estadísticas del autómata"""
        live_cells = np.sum(self.current_state)
        density = live_cells / self.width if self.width > 0 else 0
        
        return {
            "generation": self.generation,
            "live_cells": live_cells,
            "width": self.width,
            "density": round(density, 4),
            "rule": self.rule,
            "rule_binary": self.get_rule_binary(),
            "total_generations": len(self.history)
        }
    
    def reset(self):
        """Reiniciar el autómata"""
        self.current_state = np.zeros(self.width, dtype=int)
        if self.width > 0:
            self.current_state[self.width // 2] = 1
        self.generation = 0
        self.history = [self.current_state.copy()]

class RuleExplorer:
    """Explorador de reglas famosas de autómatas unidimensionales"""
    
    @staticmethod
    def get_famous_rules():
        return {
            "Regla 30 (Caos)": 30,
            "Regla 90 (Fractal)": 90,
            "Regla 110 (Universal)": 110,
            "Regla 184 (Tráfico)": 184,
            "Regla 54 (Compleja)": 54,
            "Regla 18 (Fractal)": 18,
            "Regla 22 (Compleja)": 22,
            "Regla 126 (Caótica)": 126,
            "Regla 150 (Lineal)": 150,
            "Regla 250 (Identidad)": 250
        }
    
    @staticmethod
    def get_rule_description(rule: int):
        descriptions = {
            30: "Regla caótica - Genera patrones aleatorios, usada en Mathematica",
            90: "Regla fractal - Genera el triángulo de Sierpinski",
            110: "Regla universal - Turing completa, puede simular cualquier computadora",
            184: "Regla de tráfico - Modela flujo de tráfico vehicular",
            54: "Regla compleja - Exhibe comportamiento complejo interesante",
            18: "Regla fractal - Patrones repetitivos y fractales",
            22: "Regla compleja - Comportamiento entre orden y caos",
            126: "Regla caótica - Patrones complejos y aleatorios",
            150: "Regla lineal - Comportamiento lineal y predecible",
            250: "Regla identidad - Copia el estado anterior"
        }
        return descriptions.get(rule, "Regla estándar sin descripción específica")