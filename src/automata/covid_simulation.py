"""
Simulación de pandemia COVID-19 usando autómatas celulares
"""

import numpy as np
from typing import Dict, Tuple
import random

class CovidSimulation:
    """Simulación de pandemia usando modelo SIR en autómatas celulares"""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # 0: Susceptible, 1: Infectado, 2: Recuperado
        self.day = 0
        self.history = []
        
        # Parámetros epidemiológicos
        self.infection_rate = 0.3
        self.recovery_rate = 0.1
        self.infection_radius = 1
        self.mobility = 0.1
        
    def initialize_outbreak(self, initial_infected: int = 5):
        """Inicializar brote inicial"""
        # Población inicial: todos susceptibles
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        
        # Infectar algunas células aleatorias
        infected_positions = random.sample(
            [(i, j) for i in range(self.rows) for j in range(self.cols)], 
            min(initial_infected, self.rows * self.cols)
        )
        
        for i, j in infected_positions:
            self.grid[i, j] = 1  # Infectado
        
        self.day = 0
        self.history = [self.get_statistics()]
    
    def set_parameters(self, infection_rate: float, recovery_rate: float, 
                      infection_radius: int, mobility: float):
        """Establecer parámetros de la simulación"""
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.infection_radius = infection_radius
        self.mobility = mobility
    
    def count_neighbors_infected(self, row: int, col: int) -> int:
        """Contar vecinos infectados"""
        infected_count = 0
        total_neighbors = 0
        
        for i in range(-self.infection_radius, self.infection_radius + 1):
            for j in range(-self.infection_radius, self.infection_radius + 1):
                if i == 0 and j == 0:
                    continue
                    
                r = (row + i) % self.rows
                c = (col + j) % self.cols
                
                if self.grid[r, c] == 1:  # Infectado
                    infected_count += 1
                total_neighbors += 1
        
        return infected_count
    
    def next_day(self):
        """Avanzar un día en la simulación"""
        new_grid = self.grid.copy()
        
        # Aplicar movilidad (cambiar posiciones aleatoriamente)
        if self.mobility > 0:
            for _ in range(int(self.rows * self.cols * self.mobility)):
                i1, j1 = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
                i2, j2 = random.randint(0, self.rows-1), random.randint(0, self.cols-1)
                new_grid[i1, j1], new_grid[i2, j2] = new_grid[i2, j2], new_grid[i1, j1]
        
        # Aplicar reglas epidemiológicas
        for row in range(self.rows):
            for col in range(self.cols):
                current_state = self.grid[row, col]
                
                if current_state == 0:  # Susceptible
                    infected_neighbors = self.count_neighbors_infected(row, col)
                    infection_prob = 1 - (1 - self.infection_rate) ** infected_neighbors
                    
                    if random.random() < infection_prob:
                        new_grid[row, col] = 1  # Se infecta
                
                elif current_state == 1:  # Infectado
                    if random.random() < self.recovery_rate:
                        new_grid[row, col] = 2  # Se recupera
        
        self.grid = new_grid
        self.day += 1
        self.history.append(self.get_statistics())
    
    def get_statistics(self) -> Dict:
        """Obtener estadísticas actuales"""
        total_cells = self.rows * self.cols
        susceptible = np.sum(self.grid == 0)
        infected = np.sum(self.grid == 1)
        recovered = np.sum(self.grid == 2)
        
        return {
            "day": self.day,
            "susceptible": susceptible,
            "infected": infected,
            "recovered": recovered,
            "total": total_cells,
            "susceptible_pct": round(susceptible / total_cells * 100, 2),
            "infected_pct": round(infected / total_cells * 100, 2),
            "recovered_pct": round(recovered / total_cells * 100, 2),
            "peak_infected": max([h["infected_pct"] for h in self.history]) if self.history else 0
        }
    
    def get_epidemic_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obtener curva epidémica"""
        if not self.history:
            return np.array([]), np.array([]), np.array([])
        
        days = np.array([h["day"] for h in self.history])
        susceptible = np.array([h["susceptible_pct"] for h in self.history])
        infected = np.array([h["infected_pct"] for h in self.history])
        recovered = np.array([h["recovered_pct"] for h in self.history])
        
        return days, susceptible, infected, recovered
    
    def reset(self):
        """Reiniciar simulación"""
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.day = 0
        self.history = []