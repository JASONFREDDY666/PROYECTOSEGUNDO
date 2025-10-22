"""
Juego de la Vida de Conway
"""

import numpy as np
from typing import Tuple, List
import random

class GameOfLife:
    """Implementación del Juego de la Vida de Conway"""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.generation = 0
        self.history = []
        
    def random_initialization(self, density: float = 0.3):
        """Inicialización aleatoria del tablero"""
        self.grid = np.random.choice([0, 1], size=(self.rows, self.cols), 
                                   p=[1-density, density])
        self.generation = 0
        self.history = [self.grid.copy()]
        
    def set_initial_state(self, initial_grid: np.ndarray):
        """Establecer un estado inicial específico"""
        if initial_grid.shape == (self.rows, self.cols):
            self.grid = initial_grid.copy()
            self.generation = 0
            self.history = [self.grid.copy()]
        
    def count_neighbors(self, row: int, col: int) -> int:
        """Contar vecinos vivos de una célula"""
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r = (row + i) % self.rows
                c = (col + j) % self.cols
                count += self.grid[r, c]
        return count
    
    def next_generation(self):
        """Calcular la siguiente generación"""
        new_grid = np.zeros((self.rows, self.cols), dtype=int)
        
        for row in range(self.rows):
            for col in range(self.cols):
                neighbors = self.count_neighbors(row, col)
                
                # Reglas del Juego de la Vida
                if self.grid[row, col] == 1:  # Célula viva
                    if neighbors in [2, 3]:
                        new_grid[row, col] = 1  # Sobrevive
                    else:
                        new_grid[row, col] = 0  # Muere
                else:  # Célula muerta
                    if neighbors == 3:
                        new_grid[row, col] = 1  # Nace
                    else:
                        new_grid[row, col] = 0  # Permanece muerta
        
        self.grid = new_grid
        self.generation += 1
        self.history.append(self.grid.copy())
        
    def get_live_cells_count(self) -> int:
        """Contar células vivas"""
        return np.sum(self.grid)
    
    def get_statistics(self) -> dict:
        """Obtener estadísticas del estado actual"""
        live_cells = self.get_live_cells_count()
        total_cells = self.rows * self.cols
        density = live_cells / total_cells if total_cells > 0 else 0
        
        return {
            "generation": self.generation,
            "live_cells": live_cells,
            "total_cells": total_cells,
            "density": round(density, 4),
            "extinct": live_cells == 0,
            "stable": len(self.history) > 10 and 
                     np.array_equal(self.grid, self.history[-2])
        }
    
    def reset(self):
        """Reiniciar el juego"""
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.generation = 0
        self.history = []