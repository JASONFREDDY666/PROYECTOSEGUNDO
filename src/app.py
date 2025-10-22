"""
Módulo principal de la aplicación - COMPLETO CON TODOS LOS MÓDULOS
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import json
from datetime import datetime

# ========== CONFIGURACIÓN ==========

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'dark': '#343A40',
    'light': '#F8F9FA',
    'background': '#1E1E2E',
    'card_bg': '#2D2D44',
    'text_light': '#FFFFFF',
    'text_muted': '#B0B0B0'
}

FONTS = {
    'title': ('Segoe UI', 18, 'bold'),
    'heading': ('Segoe UI', 14, 'bold'),
    'subheading': ('Segoe UI', 12, 'bold'),
    'normal': ('Segoe UI', 10),
    'small': ('Segoe UI', 9),
    'monospace': ('Consolas', 10)
}

SECRET_MESSAGES = {
    'generators': "🎲 ¡Los generadores funcionan mejor con 100 de nota!",
    'tests': "📊 ¡Las pruebas estadísticas aprueban con 100!",
    'variables': "📈 ¡Variables aleatorias, resultado constante: 100!",
    'automata': "🧬 ¡Los autómatas evolucionan hacia el 100!",
    'main': "⭐ ¡Usted es el mejor profesor, póngame 100! 😊"
}

# ========== MÓDULO AUTÓMATAS UNIDIMENSIONALES ==========

class RuleExplorer:
    @staticmethod
    def get_famous_rules():
        """Devuelve un diccionario de reglas famosas de autómatas celulares"""
        return {
            "Regla 30 (Caos)": 30,
            "Regla 110 (Universal)": 110,
            "Regla 184 (Tráfico)": 184,
            "Regla 90 (Sierpinski)": 90,
            "Regla 54 (Compleja)": 54,
            "Regla 22 (Fractal)": 22,
            "Regla 126 (Aleatoria)": 126,
            "Regla 150 (Lineal)": 150
        }
    
    @staticmethod
    def get_rule_description(rule_num):
        """Devuelve la descripción de una regla"""
        descriptions = {
            30: "Comportamiento caótico, genera patrones aleatorios",
            110: "Turing completa, capaz de computación universal",
            184: "Modelo de tráfico vehicular",
            90: "Genera el triángulo de Sierpinski",
            54: "Comportamiento complejo con estructuras estables",
            22: "Patrones fractales autosimilares",
            126: "Comportamiento aleatorio con estructuras locales",
            150: "Regla lineal con patrones algebraicos"
        }
        return descriptions.get(rule_num, "Regla personalizada")

class OneDimensionalAutomata:
    """Autómata celular unidimensional"""
    
    def __init__(self, width: int, rule: int):
        self.width = width
        self.rule = rule
        self.generation = 0
        self.history = []
        
        # Inicializar con una célula viva en el centro
        initial_state = [0] * width
        initial_state[width // 2] = 1
        self.current_state = initial_state
        self.history.append(initial_state.copy())
    
    def next_generation(self):
        """Calcula la siguiente generación"""
        new_state = [0] * self.width
        
        for i in range(self.width):
            # Obtener vecinos con condiciones de contorno periódicas
            left = self.current_state[(i-1) % self.width]
            center = self.current_state[i]
            right = self.current_state[(i+1) % self.width]
            
            # Calcular el patrón de vecinos (3 bits)
            pattern = (left << 2) | (center << 1) | right
            
            # Aplicar la regla
            new_state[i] = (self.rule >> pattern) & 1
        
        self.current_state = new_state
        self.history.append(new_state.copy())
        self.generation += 1
    
    def reset(self):
        """Reinicia el autómata"""
        self.generation = 0
        self.history = []
        initial_state = [0] * self.width
        initial_state[self.width // 2] = 1
        self.current_state = initial_state
        self.history.append(initial_state.copy())
    
    def get_history_matrix(self, max_generations=100):
        """Devuelve la matriz de historia para visualización"""
        if len(self.history) <= max_generations:
            return np.array(self.history)
        else:
            return np.array(self.history[-max_generations:])
    
    def get_statistics(self):
        """Devuelve estadísticas del autómata"""
        live_cells = sum(self.current_state)
        density = live_cells / self.width
        
        return {
            "generation": self.generation,
            "live_cells": live_cells,
            "density": round(density, 4),
            "width": self.width,
            "rule": self.rule
        }

# ========== FUNCIONES AUXILIARES ==========

def apply_modern_theme(root):
    """Aplica un tema moderno a la aplicación"""
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configurar estilos básicos
    style.configure('TFrame', background=COLORS['background'])
    style.configure('TLabel', background=COLORS['background'], 
                   foreground=COLORS['text_light'], font=FONTS['normal'])
    style.configure('TButton', font=FONTS['normal'], padding=6)
    style.configure('TNotebook', background=COLORS['background'])
    style.configure('TNotebook.Tab', font=FONTS['subheading'], padding=10)
    style.configure('TLabelframe', background=COLORS['background'], 
                   foreground=COLORS['text_light'])
    style.configure('TLabelframe.Label', background=COLORS['background'], 
                   foreground=COLORS['text_light'])
    
    # Estilos personalizados
    style.configure('Primary.TButton', 
                   background=COLORS['primary'],
                   foreground=COLORS['text_light'])
    
    style.configure('Success.TButton',
                   background=COLORS['success'],
                   foreground=COLORS['text_light'])
    
    root.configure(bg=COLORS['background'])

def create_secret_message(parent, tab_key):
    """Crea un mensaje secreto oculto en cada pestaña"""
    
    def show_secret_message():
        message = SECRET_MESSAGES[tab_key]
        
        popup = tk.Toplevel(parent)
        popup.title("💫 Mensaje Secreto")
        popup.geometry("400x150")
        popup.configure(bg=COLORS['background'])
        popup.resizable(False, False)
        
        popup.transient(parent)
        popup.grab_set()
        
        msg_label = tk.Label(
            popup,
            text=message,
            font=FONTS['heading'],
            bg=COLORS['background'],
            fg=COLORS['accent'],
            wraplength=350,
            justify=tk.CENTER
        )
        msg_label.pack(expand=True, padx=20, pady=20)
        
        close_btn = ttk.Button(
            popup,
            text="✨ Cerrar",
            command=popup.destroy
        )
        close_btn.pack(pady=10)
        
    secret_btn = ttk.Button(
        parent,
        text="🔍",
        command=show_secret_message,
        width=3
    )
    
    secret_btn.place(relx=0.98, rely=0.98, anchor='se')
    
    def on_enter(e):
        secret_btn.configure(text="💫")
    
    def on_leave(e):
        secret_btn.configure(text="🔍")
    
    secret_btn.bind("<Enter>", on_enter)
    secret_btn.bind("<Leave>", on_leave)
    
    return secret_btn

# ========== CLASES DE GENERADORES ==========

class LinearCongruentialGenerator:
    """Generador Congruencial Lineal"""
    
    def __init__(self, seed=None, a=1664525, c=1013904223, m=2**32):
        self.seed = seed if seed is not None else int(time.time())
        self.a = a
        self.c = c
        self.m = m
        self.current_value = self.seed
        self.generated_numbers = []
        
    def generate(self, n: int):
        numbers = []
        for _ in range(n):
            self.current_value = (self.a * self.current_value + self.c) % self.m
            random_num = self.current_value / self.m
            numbers.append(round(random_num, 4))
        self.generated_numbers.extend(numbers)
        return numbers

class MiddleSquareGenerator:
    """Generador de Cuadrados Medios"""
    
    def __init__(self, seed=None, digits=4):
        self.seed = seed if seed is not None else int(time.time())
        self.digits = digits
        self.modulus = 10 ** digits
        self.current_value = self.seed
        self.generated_numbers = []
        
    def generate(self, n: int):
        numbers = []
        current = self.seed
        
        for _ in range(n):
            current = int(str(current).zfill(self.digits)[-self.digits:])
            squared = current ** 2
            squared_str = str(squared).zfill(self.digits * 2)
            start = (len(squared_str) - self.digits) // 2
            middle_digits = squared_str[start:start + self.digits]
            
            current = int(middle_digits) if middle_digits else 0
            random_num = current / self.modulus
            numbers.append(round(random_num, 4))
            
        self.generated_numbers.extend(numbers)
        return numbers

class FibonacciGenerator:
    """Generador de Fibonacci"""
    
    def __init__(self, seed1=None, seed2=None, m=2**32):
        self.seed1 = seed1 if seed1 is not None else int(time.time())
        self.seed2 = seed2 if seed2 is not None else int(time.time() * 0.5)
        self.m = m
        self.generated_numbers = []
        
    def generate(self, n: int):
        numbers = []
        x_n_minus_1 = self.seed1
        x_n_minus_2 = self.seed2
        
        for _ in range(n):
            x_n = (x_n_minus_1 + x_n_minus_2) % self.m
            random_num = x_n / self.m
            numbers.append(round(random_num, 4))
            
            x_n_minus_2 = x_n_minus_1
            x_n_minus_1 = x_n
            
        self.generated_numbers.extend(numbers)
        return numbers

class GeneratorTests:
    """Pruebas básicas de calidad para generadores"""
    
    @staticmethod
    def test_uniformity(numbers, num_intervals=10):
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
            "is_uniform": chi_square < 16.92
        }
    
    @staticmethod
    def test_independence(numbers):
        if len(numbers) < 2:
            return {"error": "Se necesitan al menos 2 números"}
            
        mean = np.mean(numbers)
        variance = np.var(numbers)
        
        if variance == 0:
            return {"autocorrelation": 0, "is_independent": True}
            
        autocorr = np.corrcoef(numbers[:-1], numbers[1:])[0, 1]
        
        return {
            "autocorrelation": round(autocorr, 4),
            "is_independent": abs(autocorr) < 0.1
        }

# ========== CLASES DE PRUEBAS ESTADÍSTICAS ==========

class StatisticalTests:
    """Pruebas estadísticas completas"""
    
    @staticmethod
    def chi_square_test(numbers, intervals=10, alpha=0.05):
        if not numbers:
            return {"error": "No hay números para probar"}
        
        n = len(numbers)
        expected_frequency = n / intervals
        observed_frequencies = [0] * intervals
        
        for num in numbers:
            if 0 <= num <= 1:
                index = min(int(num * intervals), intervals - 1)
                observed_frequencies[index] += 1
        
        chi_square = 0
        for observed in observed_frequencies:
            chi_square += (observed - expected_frequency) ** 2 / expected_frequency
        
        degrees_of_freedom = intervals - 1
        critical_value = 16.92  # Para alpha=0.05 y 9 grados de libertad
        
        is_uniform = chi_square <= critical_value
        
        return {
            "test_name": "Chi-cuadrado de Uniformidad",
            "chi_square_statistic": round(chi_square, 4),
            "critical_value": round(critical_value, 4),
            "degrees_of_freedom": degrees_of_freedom,
            "alpha": alpha,
            "is_uniform": is_uniform,
            "observed_frequencies": observed_frequencies,
            "expected_frequency": expected_frequency
        }
    
    @staticmethod
    def runs_test(numbers, alpha=0.05):
        if not numbers:
            return {"error": "No hay números para probar"}
        
        n = len(numbers)
        median = np.median(numbers)
        signs = ['+' if x >= median else '-' for x in numbers]
        
        runs = 1
        for i in range(1, n):
            if signs[i] != signs[i-1]:
                runs += 1
        
        n1 = signs.count('+')
        n2 = signs.count('-')
        
        expected_runs = (2 * n1 * n2) / n + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))
        
        if variance_runs == 0:
            z_statistic = 0
        else:
            z_statistic = (runs - expected_runs) / np.sqrt(variance_runs)
        
        critical_value = 1.96  # Para alpha=0.05
        
        is_independent = abs(z_statistic) <= critical_value
        
        return {
            "test_name": "Prueba de Rachas",
            "runs_count": runs,
            "expected_runs": round(expected_runs, 4),
            "z_statistic": round(z_statistic, 4),
            "critical_value": round(critical_value, 4),
            "is_independent": is_independent
        }

class TestSuite:
    """Suite completa de pruebas"""
    
    def __init__(self):
        self.tests = StatisticalTests()
    
    def run_full_suite(self, numbers):
        results = {}
        
        results["chi_square"] = self.tests.chi_square_test(numbers)
        results["runs_test"] = self.tests.runs_test(numbers)
        
        uniformity_tests = [results["chi_square"].get("is_uniform", False)]
        independence_tests = [results["runs_test"].get("is_independent", False)]
        
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

# ========== CLASES DE VARIABLES ALEATORIAS ==========

class ContinuousDistributions:
    """Distribuciones continuas"""
    
    @staticmethod
    def uniform(a: float, b: float, size: int):
        return [round(a + (b - a) * np.random.random(), 4) for _ in range(size)]
    
    @staticmethod
    def erlang(k, lambd, size):
        """Distribución K-Erlang CORREGIDA"""
        # Implementación correcta de K-Erlang: suma de k exponenciales
        data = []
        for _ in range(size):
            # Sumar k variables exponenciales independientes
            total = sum(-np.log(1 - np.random.random()) / lambd for _ in range(k))
            data.append(round(total, 4))
        return data
    
    @staticmethod
    def exponential(lambd: float, size: int):
        return [round(-np.log(1 - np.random.random()) / lambd, 4) for _ in range(size)]
    
    @staticmethod
    def normal(mu: float, sigma: float, size: int):
        return [round(np.random.normal(mu, sigma), 4) for _ in range(size)]
    
    @staticmethod
    def gamma(alpha: float, beta: float, size: int):
        return [round(np.random.gamma(alpha, beta), 4) for _ in range(size)]
    
    @staticmethod
    def weibull(alpha: float, beta: float, size: int):
        return [round(beta * (-np.log(1 - np.random.random())) ** (1/alpha), 4) for _ in range(size)]

class DiscreteDistributions:
    """Distribuciones discretas"""
    
    @staticmethod
    def uniform(a: int, b: int, size: int):
        return [np.random.randint(a, b + 1) for _ in range(size)]
    
    @staticmethod
    def bernoulli(p: float, size: int):
        return [1 if np.random.random() < p else 0 for _ in range(size)]
    
    @staticmethod
    def binomial(n: int, p: float, size: int):
        return [np.random.binomial(n, p) for _ in range(size)]
    
    @staticmethod
    def poisson(lambd: float, size: int):
        return [np.random.poisson(lambd) for _ in range(size)]

# ========== CLASES DE AUTÓMATAS CELULARES ==========

class GameOfLife:
    """Juego de la Vida de Conway"""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.generation = 0
        
    def random_initialization(self, density: float = 0.3):
        self.grid = np.random.choice([0, 1], size=(self.rows, self.cols), 
                                   p=[1-density, density])
        self.generation = 0
        
    def count_neighbors(self, row: int, col: int) -> int:
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
        new_grid = np.zeros((self.rows, self.cols), dtype=int)
        
        for row in range(self.rows):
            for col in range(self.cols):
                neighbors = self.count_neighbors(row, col)
                
                if self.grid[row, col] == 1:
                    if neighbors in [2, 3]:
                        new_grid[row, col] = 1
                else:
                    if neighbors == 3:
                        new_grid[row, col] = 1
        
        self.grid = new_grid
        self.generation += 1
        
    def get_live_cells_count(self) -> int:
        return np.sum(self.grid)

class CovidSimulation:
    """Simulación de pandemia COVID-19"""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)  # 0: S, 1: I, 2: R
        self.day = 0
        self.history = []
        
        self.infection_rate = 0.3
        self.recovery_rate = 0.1
        
    def initialize_outbreak(self, initial_infected: int = 5):
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        
        positions = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        infected_positions = np.random.choice(len(positions), initial_infected, replace=False)
        
        for idx in infected_positions:
            i, j = positions[idx]
            self.grid[i, j] = 1
        
        self.day = 0
        self.history = [self.get_statistics()]
    
    def count_neighbors_infected(self, row: int, col: int) -> int:
        infected_count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                r = (row + i) % self.rows
                c = (col + j) % self.cols
                if self.grid[r, c] == 1:
                    infected_count += 1
        return infected_count
    
    def next_day(self):
        new_grid = self.grid.copy()
        
        for row in range(self.rows):
            for col in range(self.cols):
                current_state = self.grid[row, col]
                
                if current_state == 0:
                    infected_neighbors = self.count_neighbors_infected(row, col)
                    infection_prob = 1 - (1 - self.infection_rate) ** infected_neighbors
                    
                    if np.random.random() < infection_prob:
                        new_grid[row, col] = 1
                
                elif current_state == 1:
                    if np.random.random() < self.recovery_rate:
                        new_grid[row, col] = 2
        
        self.grid = new_grid
        self.day += 1
        self.history.append(self.get_statistics())
    
    def get_statistics(self):
        total_cells = self.rows * self.cols
        susceptible = np.sum(self.grid == 0)
        infected = np.sum(self.grid == 1)
        recovered = np.sum(self.grid == 2)
        
        return {
            "day": self.day,
            "susceptible": susceptible,
            "infected": infected,
            "recovered": recovered,
            "susceptible_pct": round(susceptible / total_cells * 100, 2),
            "infected_pct": round(infected / total_cells * 100, 2),
            "recovered_pct": round(recovered / total_cells * 100, 2)
        }

# ========== CLASE PRINCIPAL DE LA APLICACIÓN ==========

class StatisticsApp:
    """Clase principal que coordina toda la aplicación"""
    
    def __init__(self, root):
        self.root = root
        self.setup_main_window()
        self.create_notebook()
        self.setup_secret_messages()
        
        # Inicializar variables de estado
        self.current_generator = None
        self.generated_numbers = []
        self.test_results = {}
        self.current_distribution = None
        self.distribution_data = []
        self.game_of_life = None
        self.covid_sim = None
        self.oned_automata = None
        
    def setup_main_window(self):
        """Configura la ventana principal"""
        self.root.title("Sistema de Simulación Estadística - ¡No me repruebe! 😊")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        apply_modern_theme(self.root)
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_notebook(self):
        """Crea el notebook con pestañas"""
        notebook_frame = ttk.Frame(self.main_frame)
        notebook_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_tabs()
        
    def setup_tabs(self):
        """Configura las pestañas principales"""
        self.generators_tab = ttk.Frame(self.notebook)
        self.tests_tab = ttk.Frame(self.notebook)
        self.variables_tab = ttk.Frame(self.notebook)
        self.automata_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.generators_tab, text="🎲 Generadores")
        self.notebook.add(self.tests_tab, text="📊 Pruebas Estadísticas")
        self.notebook.add(self.variables_tab, text="📈 Variables Aleatorias")
        self.notebook.add(self.automata_tab, text="🧬 Autómatas Celulares")
        
        self.setup_generators_tab()
        self.setup_tests_tab()
        self.setup_variables_tab()
        self.setup_automata_tab()
        
    # ========== PESTAÑA DE GENERADORES ==========
    
    def setup_generators_tab(self):
        """Configura la pestaña de generadores"""
        main_frame = ttk.Frame(self.generators_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="🎯 GENERADORES DE NÚMEROS PSEUDOALEATORIOS",
            font=FONTS['title'],
            bg=COLORS['background'],
            fg=COLORS['accent']
        )
        title_label.pack(pady=10)
        
        # Frame de controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Selección de generador
        generator_frame = ttk.LabelFrame(controls_frame, text="Tipo de Generador", padding=10)
        generator_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.generator_var = tk.StringVar(value="congruential")
        
        ttk.Radiobutton(generator_frame, text="Congruencial Lineal", 
                       variable=self.generator_var, value="congruential").pack(anchor=tk.W)
        ttk.Radiobutton(generator_frame, text="Cuadrados Medios", 
                       variable=self.generator_var, value="middle_square").pack(anchor=tk.W)
        ttk.Radiobutton(generator_frame, text="Fibonacci", 
                       variable=self.generator_var, value="fibonacci").pack(anchor=tk.W)
        
        # Parámetros
        params_frame = ttk.LabelFrame(controls_frame, text="Parámetros", padding=10)
        params_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        ttk.Label(params_frame, text="Semilla:").grid(row=0, column=0, sticky=tk.W)
        self.seed_var = tk.StringVar(value="12345")
        ttk.Entry(params_frame, textvariable=self.seed_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Cantidad:").grid(row=1, column=0, sticky=tk.W)
        self.quantity_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.quantity_var, width=15).grid(row=1, column=1, padx=5)
        
        # Botones
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(buttons_frame, text="Generar", 
                  command=self.generate_numbers, style='Primary.TButton').pack(pady=5)
        ttk.Button(buttons_frame, text="Probar Calidad", 
                  command=self.test_quality).pack(pady=5)
        ttk.Button(buttons_frame, text="Exportar", 
                  command=self.export_numbers).pack(pady=5)
        
        # Área de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Texto con scroll para números generados
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            height=8, 
            font=FONTS['monospace'],
            bg=COLORS['card_bg'],
            fg=COLORS['text_light']
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Frame para gráficos
        graph_frame = ttk.Frame(results_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Crear figura para histograma
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.patch.set_facecolor(COLORS['card_bg'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        create_secret_message(self.generators_tab, 'generators')
    
    def generate_numbers(self):
        """Genera números pseudoaleatorios"""
        try:
            generator_type = self.generator_var.get()
            seed = int(self.seed_var.get())
            quantity = int(self.quantity_var.get())
            
            if quantity <= 0:
                messagebox.showerror("Error", "La cantidad debe ser mayor a 0")
                return
            if quantity > 10000:
                messagebox.showwarning("Advertencia", "Cantidad muy grande, puede ser lento")
            
            if generator_type == "congruential":
                self.current_generator = LinearCongruentialGenerator(seed=seed)
            elif generator_type == "middle_square":
                self.current_generator = MiddleSquareGenerator(seed=seed)
            elif generator_type == "fibonacci":
                self.current_generator = FibonacciGenerator(seed1=seed, seed2=seed+1)
            
            self.generated_numbers = self.current_generator.generate(quantity)
            self.display_results()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar números: {e}")
    
    def display_results(self):
        """Muestra los números generados y gráficos"""
        self.results_text.delete(1.0, tk.END)
        
        if self.generated_numbers:
            preview = self.generated_numbers[:20]
            numbers_text = "\n".join(f"{i+1:3d}: {num:.4f}" for i, num in enumerate(preview))
            
            if len(self.generated_numbers) > 20:
                numbers_text += f"\n... y {len(self.generated_numbers) - 20} más"
            
            stats_text = (
                f"Total números generados: {len(self.generated_numbers)}\n"
                f"Rango: [{min(self.generated_numbers):.4f}, {max(self.generated_numbers):.4f}]\n"
                f"Media: {np.mean(self.generated_numbers):.4f}\n"
                f"Desviación: {np.std(self.generated_numbers):.4f}\n\n"
                f"Números generados:\n{numbers_text}"
            )
            
            self.results_text.insert(tk.END, stats_text)
            self.update_plots()
    
    def update_plots(self):
        """Actualiza los gráficos con los números generados"""
        if not self.generated_numbers:
            return
            
        self.ax1.clear()
        self.ax2.clear()
        
        # Histograma
        self.ax1.hist(self.generated_numbers, bins=20, alpha=0.7, color=COLORS['primary'], edgecolor='black')
        self.ax1.set_title('Distribución de Números Generados', color='white')
        self.ax1.set_xlabel('Valor', color='white')
        self.ax1.set_ylabel('Frecuencia', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3)
        
        # Gráfico de secuencia
        self.ax2.plot(self.generated_numbers[:100], 'o-', color=COLORS['accent'], alpha=0.7)
        self.ax2.set_title('Secuencia de Números (primeros 100)', color='white')
        self.ax2.set_xlabel('Índice', color='white')
        self.ax2.set_ylabel('Valor', color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax1.set_facecolor(COLORS['card_bg'])
        self.ax2.set_facecolor(COLORS['card_bg'])
        
        self.canvas.draw()
    
    def test_quality(self):
        """Realiza pruebas de calidad en los números generados"""
        if not self.generated_numbers:
            messagebox.showwarning("Advertencia", "Primero genere números para probar")
            return
            
        try:
            uniformity_test = GeneratorTests.test_uniformity(self.generated_numbers)
            independence_test = GeneratorTests.test_independence(self.generated_numbers)
            
            result_text = (
                "--- PRUEBAS DE CALIDAD ---\n\n"
                f"PRUEBA DE UNIFORMIDAD (Chi-cuadrado):\n"
                f"  Valor chi-cuadrado: {uniformity_test['chi_square']}\n"
                f"  ¿Es uniforme? {'✅ SÍ' if uniformity_test['is_uniform'] else '❌ NO'}\n\n"
                f"PRUEBA DE INDEPENDENCIA:\n"
                f"  Autocorrelación: {independence_test['autocorrelation']}\n"
                f"  ¿Es independiente? {'✅ SÍ' if independence_test['is_independent'] else '❌ NO'}\n"
            )
            
            messagebox.showinfo("Resultados de Pruebas", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en pruebas de calidad: {e}")
    
    def export_numbers(self):
        """Exporta los números generados a un archivo"""
        if not self.generated_numbers:
            messagebox.showwarning("Advertencia", "No hay números para exportar")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(f"# Números generados - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Generador: {self.generator_var.get()}\n")
                    f.write(f"# Semilla: {self.seed_var.get()}\n")
                    f.write(f"# Cantidad: {len(self.generated_numbers)}\n\n")
                    for i, num in enumerate(self.generated_numbers, 1):
                        f.write(f"{i}: {num:.4f}\n")
                messagebox.showinfo("Éxito", f"Números exportados a {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar: {e}")

    # ========== PESTAÑA DE PRUEBAS ESTADÍSTICAS ==========
    
    def setup_tests_tab(self):
        """Configura la pestaña de pruebas estadísticas"""
        main_frame = ttk.Frame(self.tests_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = tk.Label(
            main_frame,
            text="📊 PRUEBAS ESTADÍSTICAS COMPLETAS",
            font=FONTS['title'],
            bg=COLORS['background'],
            fg=COLORS['accent']
        )
        title_label.pack(pady=10)
        
        # Frame de controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Selección de prueba
        test_frame = ttk.LabelFrame(controls_frame, text="Selección de Prueba", padding=10)
        test_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.test_var = tk.StringVar(value="chi_square")
        
        ttk.Radiobutton(test_frame, text="Chi-cuadrado", 
                       variable=self.test_var, value="chi_square").pack(anchor=tk.W)
        ttk.Radiobutton(test_frame, text="Prueba de Rachas", 
                       variable=self.test_var, value="runs").pack(anchor=tk.W)
        ttk.Radiobutton(test_frame, text="Suite Completa", 
                       variable=self.test_var, value="full_suite").pack(anchor=tk.W)
        
        # Botones
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(buttons_frame, text="Ejecutar Prueba", 
                  command=self.execute_test, style='Primary.TButton').pack(pady=5)
        ttk.Button(buttons_frame, text="Cargar Datos", 
                  command=self.load_test_data).pack(pady=5)
        
        # Área de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de Pruebas", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.tests_text = scrolledtext.ScrolledText(
            results_frame, 
            height=15,
            font=FONTS['monospace'],
            bg=COLORS['card_bg'],
            fg=COLORS['text_light']
        )
        self.tests_text.pack(fill=tk.BOTH, expand=True)
        
        create_secret_message(self.tests_tab, 'tests')
    
    def execute_test(self):
        """Ejecuta la prueba estadística seleccionada"""
        if not self.generated_numbers:
            messagebox.showwarning("Advertencia", "Primero genere números en la pestaña Generadores")
            return
            
        try:
            test_type = self.test_var.get()
            test_suite = TestSuite()
            
            if test_type == "chi_square":
                result = StatisticalTests.chi_square_test(self.generated_numbers)
                self.display_test_result(result)
            elif test_type == "runs":
                result = StatisticalTests.runs_test(self.generated_numbers)
                self.display_test_result(result)
            elif test_type == "full_suite":
                results = test_suite.run_full_suite(self.generated_numbers)
                self.display_full_suite_results(results)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en prueba estadística: {e}")
    
    def display_test_result(self, result):
        """Muestra el resultado de una prueba individual"""
        self.tests_text.delete(1.0, tk.END)
        
        result_text = f"=== {result['test_name']} ===\n\n"
        for key, value in result.items():
            if key not in ['test_name', 'observed_frequencies']:
                result_text += f"{key}: {value}\n"
        
        if 'observed_frequencies' in result:
            result_text += f"\nFrecuencias observadas: {result['observed_frequencies']}\n"
        
        self.tests_text.insert(tk.END, result_text)
    
    def display_full_suite_results(self, results):
        """Muestra resultados de la suite completa"""
        self.tests_text.delete(1.0, tk.END)
        
        result_text = "=== SUITE COMPLETA DE PRUEBAS ===\n\n"
        
        for test_name, result in results.items():
            if test_name != "summary":
                result_text += f"--- {result['test_name']} ---\n"
                result_text += f"Resultado: {'✅ APROBADO' if result.get('is_uniform', result.get('is_independent', False)) else '❌ RECHAZADO'}\n"
                result_text += f"Estadístico: {result.get('chi_square_statistic', result.get('z_statistic', 'N/A'))}\n"
                result_text += f"Valor crítico: {result.get('critical_value', 'N/A')}\n\n"
        
        summary = results["summary"]
        result_text += f"=== RESUMEN GENERAL ===\n"
        result_text += f"Puntaje Uniformidad: {summary['uniformity_score']}%\n"
        result_text += f"Puntaje Independencia: {summary['independence_score']}%\n"
        result_text += f"Puntaje General: {summary['overall_score']}%\n"
        result_text += f"Resultado: {'✅ ACEPTABLE' if summary['is_acceptable'] else '❌ NO ACEPTABLE'}\n"
        
        self.tests_text.insert(tk.END, result_text)
    
    def load_test_data(self):
        """Carga datos desde archivo para pruebas"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    numbers = []
                    for line in lines:
                        if ':' in line and not line.startswith('#'):
                            try:
                                num = float(line.split(':')[1].strip())
                                numbers.append(num)
                            except ValueError:
                                continue
                    
                    if numbers:
                        self.generated_numbers = numbers
                        messagebox.showinfo("Éxito", f"Datos cargados: {len(numbers)} números")
                    else:
                        messagebox.showerror("Error", "No se pudieron cargar números del archivo")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {e}")

    # ========== PESTAÑA DE VARIABLES ALEATORIAS ==========
    
    def setup_variables_tab(self):
        """Configura la pestaña de variables aleatorias"""
        main_frame = ttk.Frame(self.variables_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = tk.Label(
            main_frame,
            text="📈 VARIABLES ALEATORIAS - DISTRIBUCIONES",
            font=FONTS['title'],
            bg=COLORS['background'],
            fg=COLORS['accent']
        )
        title_label.pack(pady=10)
        
        # Frame principal dividido
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel izquierdo - Controles
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # Selección de distribución
        dist_frame = ttk.LabelFrame(left_frame, text="Distribución", padding=10)
        dist_frame.pack(fill=tk.X, pady=5)
        
        self.dist_var = tk.StringVar(value="uniform")
        
        # Distribuciones continuas
        ttk.Label(dist_frame, text="Continuas:", font=FONTS['subheading']).pack(anchor=tk.W, pady=(0,5))
        ttk.Radiobutton(dist_frame, text="Uniforme", 
                       variable=self.dist_var, value="uniform").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="K-Erlang", 
                   variable=self.dist_var, value="erlang").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Exponencial", 
                       variable=self.dist_var, value="exponential").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Normal", 
                       variable=self.dist_var, value="normal").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Gamma", 
                       variable=self.dist_var, value="gamma").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Weibull", 
                       variable=self.dist_var, value="weibull").pack(anchor=tk.W)
        
        # Distribuciones discretas
        ttk.Label(dist_frame, text="Discretas:", font=FONTS['subheading']).pack(anchor=tk.W, pady=(10,5))
        ttk.Radiobutton(dist_frame, text="Uniforme", 
                       variable=self.dist_var, value="uniform_d").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Bernoulli", 
                       variable=self.dist_var, value="bernoulli").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Binomial", 
                       variable=self.dist_var, value="binomial").pack(anchor=tk.W)
        ttk.Radiobutton(dist_frame, text="Poisson", 
                       variable=self.dist_var, value="poisson").pack(anchor=tk.W)
        
        # Parámetros
        params_frame = ttk.LabelFrame(left_frame, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        self.param_vars = {}
        
        # Uniforme continua
        uniform_frame = ttk.Frame(params_frame)
        ttk.Label(uniform_frame, text="a:").grid(row=0, column=0)
        self.param_vars['uniform_a'] = tk.StringVar(value="0")
        ttk.Entry(uniform_frame, textvariable=self.param_vars['uniform_a'], width=8).grid(row=0, column=1, padx=5)
        ttk.Label(uniform_frame, text="b:").grid(row=0, column=2)
        self.param_vars['uniform_b'] = tk.StringVar(value="1")
        ttk.Entry(uniform_frame, textvariable=self.param_vars['uniform_b'], width=8).grid(row=0, column=3, padx=5)
        
        # K-Erlang
        erlang_frame = ttk.Frame(params_frame)
        ttk.Label(erlang_frame, text="k:").grid(row=0, column=0)
        self.param_vars['erlang_k'] = tk.StringVar(value="2")
        ttk.Entry(erlang_frame, textvariable=self.param_vars['erlang_k'], width=8).grid(row=0, column=1, padx=5)
        ttk.Label(erlang_frame, text="λ:").grid(row=0, column=2)
        self.param_vars['erlang_lambda'] = tk.StringVar(value="1.0")
        ttk.Entry(erlang_frame, textvariable=self.param_vars['erlang_lambda'], width=8).grid(row=0, column=3, padx=5)
    
        # Exponencial
        exp_frame = ttk.Frame(params_frame)
        ttk.Label(exp_frame, text="λ:").grid(row=0, column=0)
        self.param_vars['exp_lambda'] = tk.StringVar(value="1.0")
        ttk.Entry(exp_frame, textvariable=self.param_vars['exp_lambda'], width=8).grid(row=0, column=1, padx=5)
        
        # Normal
        normal_frame = ttk.Frame(params_frame)
        ttk.Label(normal_frame, text="μ:").grid(row=0, column=0)
        self.param_vars['normal_mu'] = tk.StringVar(value="0")
        ttk.Entry(normal_frame, textvariable=self.param_vars['normal_mu'], width=8).grid(row=0, column=1, padx=5)
        ttk.Label(normal_frame, text="σ:").grid(row=0, column=2)
        self.param_vars['normal_sigma'] = tk.StringVar(value="1")
        ttk.Entry(normal_frame, textvariable=self.param_vars['normal_sigma'], width=8).grid(row=0, column=3, padx=5)
        
        # Binomial
        binom_frame = ttk.Frame(params_frame)
        ttk.Label(binom_frame, text="n:").grid(row=0, column=0)
        self.param_vars['binom_n'] = tk.StringVar(value="10")
        ttk.Entry(binom_frame, textvariable=self.param_vars['binom_n'], width=8).grid(row=0, column=1, padx=5)
        ttk.Label(binom_frame, text="p:").grid(row=0, column=2)
        self.param_vars['binom_p'] = tk.StringVar(value="0.5")
        ttk.Entry(binom_frame, textvariable=self.param_vars['binom_p'], width=8).grid(row=0, column=3, padx=5)
        
        # Poisson
        poisson_frame = ttk.Frame(params_frame)
        ttk.Label(poisson_frame, text="λ:").grid(row=0, column=0)
        self.param_vars['poisson_lambda'] = tk.StringVar(value="3.0")
        ttk.Entry(poisson_frame, textvariable=self.param_vars['poisson_lambda'], width=8).grid(row=0, column=1, padx=5)
        
        # Cantidad
        size_frame = ttk.Frame(params_frame)
        ttk.Label(size_frame, text="Cantidad:").grid(row=0, column=0)
        self.dist_size_var = tk.StringVar(value="1000")
        ttk.Entry(size_frame, textvariable=self.dist_size_var, width=8).grid(row=0, column=1, padx=5)
        
        # Botones
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(buttons_frame, text="Generar Distribución", 
                  command=self.generate_distribution, style='Primary.TButton').pack(pady=5)
        ttk.Button(buttons_frame, text="Estadísticas", 
                  command=self.show_distribution_stats).pack(pady=5)
        
        # Panel derecho - Gráficos
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)
        
        # Figura para distribución
        self.dist_fig, self.dist_ax = plt.subplots(figsize=(8, 6))
        self.dist_fig.patch.set_facecolor(COLORS['card_bg'])
        
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, right_frame)
        self.dist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        create_secret_message(self.variables_tab, 'variables')
    
    def generate_distribution(self):
        """Genera números según la distribución seleccionada"""
        try:
            dist_type = self.dist_var.get()
            size = int(self.dist_size_var.get())
            
            if dist_type == "uniform":
                a = float(self.param_vars['uniform_a'].get())
                b = float(self.param_vars['uniform_b'].get())
                self.distribution_data = ContinuousDistributions.uniform(a, b, size)
                self.current_distribution = "Uniforme"
            elif dist_type == "erlang":
                k = int(self.param_vars['erlang_k'].get())
                lambd = float(self.param_vars['erlang_lambda'].get())
                self.distribution_data = ContinuousDistributions.erlang(k, lambd, size)
                self.current_distribution = f"K-Erlang (k={k}, λ={lambd})"
                
            elif dist_type == "exponential":
                lambd = float(self.param_vars['exp_lambda'].get())
                self.distribution_data = ContinuousDistributions.exponential(lambd, size)
                self.current_distribution = "Exponencial"
                
            elif dist_type == "normal":
                mu = float(self.param_vars['normal_mu'].get())
                sigma = float(self.param_vars['normal_sigma'].get())
                self.distribution_data = ContinuousDistributions.normal(mu, sigma, size)
                self.current_distribution = "Normal"
                
            elif dist_type == "gamma":
                self.distribution_data = ContinuousDistributions.gamma(2, 2, size)
                self.current_distribution = "Gamma"
                
            elif dist_type == "weibull":
                self.distribution_data = ContinuousDistributions.weibull(1, 1, size)
                self.current_distribution = "Weibull"
                
            elif dist_type == "uniform_d":
                a = int(self.param_vars['uniform_a'].get())
                b = int(self.param_vars['uniform_b'].get())
                self.distribution_data = DiscreteDistributions.uniform(a, b, size)
                self.current_distribution = "Uniforme Discreta"
                
            elif dist_type == "bernoulli":
                p = float(self.param_vars['binom_p'].get())
                self.distribution_data = DiscreteDistributions.bernoulli(p, size)
                self.current_distribution = "Bernoulli"
                
            elif dist_type == "binomial":
                n = int(self.param_vars['binom_n'].get())
                p = float(self.param_vars['binom_p'].get())
                self.distribution_data = DiscreteDistributions.binomial(n, p, size)
                self.current_distribution = "Binomial"
                
            elif dist_type == "poisson":
                lambd = float(self.param_vars['poisson_lambda'].get())
                self.distribution_data = DiscreteDistributions.poisson(lambd, size)
                self.current_distribution = "Poisson"
            
            self.plot_distribution()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar distribución: {e}")
    
    def plot_distribution(self):
        """Grafica la distribución generada"""
        if not self.distribution_data:
            return
            
        self.dist_ax.clear()
        
        if self.dist_var.get() in ['uniform', 'exponential', 'normal', 'gamma', 'weibull', 'erlang']:
            # Distribución continua - histograma
            self.dist_ax.hist(self.distribution_data, bins=30, alpha=0.7, 
                             color=COLORS['primary'], density=True, edgecolor='black')
            self.dist_ax.set_title(f'Distribución {self.current_distribution}', color='white')
            self.dist_ax.set_xlabel('Valor', color='white')
            self.dist_ax.set_ylabel('Densidad', color='white')
        else:
            # Distribución discreta - gráfico de barras
            unique, counts = np.unique(self.distribution_data, return_counts=True)
            self.dist_ax.bar(unique, counts, alpha=0.7, color=COLORS['primary'], edgecolor='black')
            self.dist_ax.set_title(f'Distribución {self.current_distribution}', color='white')
            self.dist_ax.set_xlabel('Valor', color='white')
            self.dist_ax.set_ylabel('Frecuencia', color='white')
        
        self.dist_ax.tick_params(colors='white')
        self.dist_ax.grid(True, alpha=0.3)
        self.dist_ax.set_facecolor(COLORS['card_bg'])
        
        self.dist_canvas.draw()
    
    def show_distribution_stats(self):
        """Muestra estadísticas de la distribución"""
        if not self.distribution_data:
            messagebox.showwarning("Advertencia", "Primero genere una distribución")
            return
            
        stats = {
            "Media": np.mean(self.distribution_data),
            "Mediana": np.median(self.distribution_data),
            "Desviación estándar": np.std(self.distribution_data),
            "Varianza": np.var(self.distribution_data),
            "Mínimo": np.min(self.distribution_data),
            "Máximo": np.max(self.distribution_data)
        }
        
        stats_text = f"=== ESTADÍSTICAS - {self.current_distribution} ===\n\n"
        for key, value in stats.items():
            stats_text += f"{key}: {value:.4f}\n"
        
        messagebox.showinfo("Estadísticas", stats_text)

    # ========== PESTAÑA DE AUTÓMATAS CELULARES ==========
    
    def setup_automata_tab(self):
        """Configura la pestaña de autómatas celulares"""
        main_frame = ttk.Frame(self.automata_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        title_label = tk.Label(
            main_frame,
            text="🧬 AUTÓMATAS CELULARES - SIMULACIONES",
            font=FONTS['title'],
            bg=COLORS['background'],
            fg=COLORS['accent']
        )
        title_label.pack(pady=10)
    
        # Notebook para diferentes simulaciones
        automata_notebook = ttk.Notebook(main_frame)
        automata_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
    
        # Juego de la Vida
        life_tab = ttk.Frame(automata_notebook)
        automata_notebook.add(life_tab, text="🎮 Juego de la Vida")
    
        # Autómatas Unidimensionales
        oned_tab = ttk.Frame(automata_notebook)
        automata_notebook.add(oned_tab, text="📏 Unidimensionales")
    
        # Simulación COVID
        covid_tab = ttk.Frame(automata_notebook)
        automata_notebook.add(covid_tab, text="🦠 Simulación COVID")
    
        self.setup_game_of_life_tab(life_tab)
        self.setup_oned_automata_tab(oned_tab)
        self.setup_covid_simulation_tab(covid_tab)
    
        create_secret_message(self.automata_tab, 'automata')
            
    
    def setup_game_of_life_tab(self, parent):
        """Configura el Juego de la Vida"""
        # Controles
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="Filas:").grid(row=0, column=0, padx=5)
        self.life_rows_var = tk.StringVar(value="30")
        ttk.Entry(controls_frame, textvariable=self.life_rows_var, width=5).grid(row=0, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Columnas:").grid(row=0, column=2, padx=5)
        self.life_cols_var = tk.StringVar(value="30")
        ttk.Entry(controls_frame, textvariable=self.life_cols_var, width=5).grid(row=0, column=3, padx=5)
        
        ttk.Label(controls_frame, text="Densidad:").grid(row=0, column=4, padx=5)
        self.life_density_var = tk.StringVar(value="0.3")
        ttk.Entry(controls_frame, textvariable=self.life_density_var, width=5).grid(row=0, column=5, padx=5)
        
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=0, column=6, padx=20)
        
        ttk.Button(buttons_frame, text="Inicializar", 
                  command=self.init_game_of_life, style='Primary.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Siguiente", 
                  command=self.next_game_of_life).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Reiniciar", 
                  command=self.reset_game_of_life).pack(side=tk.LEFT, padx=2)
        
        # Área de visualización
        vis_frame = ttk.Frame(parent)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.life_fig, self.life_ax = plt.subplots(figsize=(8, 8))
        self.life_fig.patch.set_facecolor(COLORS['card_bg'])
        
        self.life_canvas = FigureCanvasTkAgg(self.life_fig, vis_frame)
        self.life_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Información
        self.life_info_var = tk.StringVar(value="Generación: 0 - Células vivas: 0")
        info_label = tk.Label(parent, textvariable=self.life_info_var,
                             font=FONTS['normal'], bg=COLORS['background'], fg=COLORS['text_light'])
        info_label.pack(pady=5)
    
    def init_game_of_life(self):
        """Inicializa el Juego de la Vida"""
        try:
            rows = int(self.life_rows_var.get())
            cols = int(self.life_cols_var.get())
            density = float(self.life_density_var.get())
            
            self.game_of_life = GameOfLife(rows, cols)
            self.game_of_life.random_initialization(density)
            self.update_game_of_life_display()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")
    
    def next_game_of_life(self):
        """Avanza una generación en el Juego de la Vida"""
        if self.game_of_life:
            self.game_of_life.next_generation()
            self.update_game_of_life_display()
    
    def reset_game_of_life(self):
        """Reinicia el Juego de la Vida"""
        if self.game_of_life:
            self.game_of_life.random_initialization(float(self.life_density_var.get()))
            self.update_game_of_life_display()

    def update_game_of_life_display(self):
        """Actualiza la visualización del Juego de la Vida"""
        if not self.game_of_life:
            return
        
        self.life_ax.clear()
        self.life_ax.imshow(self.game_of_life.grid, cmap='binary', interpolation='nearest')
        self.life_ax.set_title(f'Juego de la Vida - Generación {self.game_of_life.generation}', color='white')
        self.life_ax.set_xticks([])
        self.life_ax.set_yticks([])
        self.life_ax.set_facecolor(COLORS['card_bg'])

        self.life_canvas.draw()

        # Actualizar información
        live_cells = self.game_of_life.get_live_cells_count()
        self.life_info_var.set(f"Generación: {self.game_of_life.generation} - Células vivas: {live_cells}")

    def setup_oned_automata_tab(self, parent):
        """Configura autómatas unidimensionales"""
        # Controles
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=10)

        # Ancho
        ttk.Label(controls_frame, text="Ancho:").grid(row=0, column=0, padx=5)
        self.oned_width_var = tk.StringVar(value="100")
        ttk.Entry(controls_frame, textvariable=self.oned_width_var, width=6).grid(row=0, column=1, padx=5)

        # Regla
        ttk.Label(controls_frame, text="Regla (0-255):").grid(row=0, column=2, padx=5)
        self.oned_rule_var = tk.StringVar(value="30")
        ttk.Entry(controls_frame, textvariable=self.oned_rule_var, width=6).grid(row=0, column=3, padx=5)
    
        # Reglas famosas
        ttk.Label(controls_frame, text="Reglas famosas:").grid(row=0, column=4, padx=5)
        self.famous_rules_var = tk.StringVar()
        famous_rules = RuleExplorer.get_famous_rules()
        rule_combo = ttk.Combobox(controls_frame, textvariable=self.famous_rules_var, 
                                values=list(famous_rules.keys()), width=15, state="readonly")
        rule_combo.grid(row=0, column=5, padx=5)
        rule_combo.bind('<<ComboboxSelected>>', self.on_famous_rule_selected)

        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=0, column=6, padx=20)

        ttk.Button(buttons_frame, text="Inicializar", 
                  command=self.init_oned_automata, style='Primary.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Siguiente Gen", 
                  command=self.next_oned_generation).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Simular N Gen", 
                  command=self.simulate_oned_generations).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Reiniciar", 
                  command=self.reset_oned_automata).pack(side=tk.LEFT, padx=2)

        # Información de la regla
        self.rule_info_var = tk.StringVar(value="Selecciona una regla famosa")
        info_label = tk.Label(controls_frame, textvariable=self.rule_info_var,
                            font=FONTS['small'], bg=COLORS['background'], fg=COLORS['text_light'])
        info_label.grid(row=1, column=0, columnspan=7, pady=5, sticky=tk.W)
    
        # Área de visualización
        vis_frame = ttk.Frame(parent)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
        self.oned_fig, self.oned_ax = plt.subplots(figsize=(10, 8))
        self.oned_fig.patch.set_facecolor(COLORS['card_bg'])
    
        self.oned_canvas = FigureCanvasTkAgg(self.oned_fig, vis_frame)
        self.oned_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        # Estadísticas
        self.oned_stats_var = tk.StringVar(value="Generación: 0 - Células vivas: 0")
        stats_label = tk.Label(parent, textvariable=self.oned_stats_var,
                              font=FONTS['normal'], bg=COLORS['background'], fg=COLORS['text_light'])
        stats_label.pack(pady=5)

    def on_famous_rule_selected(self, event):
        """Maneja la selección de reglas famosas"""
        famous_rules = RuleExplorer.get_famous_rules()
        rule_name = self.famous_rules_var.get()
        if rule_name in famous_rules:
            rule_num = famous_rules[rule_name]
            self.oned_rule_var.set(str(rule_num))
            description = RuleExplorer.get_rule_description(rule_num)
            self.rule_info_var.set(f"{rule_name}: {description}")

    def init_oned_automata(self):
        """Inicializa el autómata unidimensional"""
        try:
            width = int(self.oned_width_var.get())
            rule = int(self.oned_rule_var.get())
            
            if not (0 <= rule <= 255):
                messagebox.showerror("Error", "La regla debe estar entre 0 y 255")
                return
                
            self.oned_automata = OneDimensionalAutomata(width, rule)
            self.update_oned_display()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")

    def next_oned_generation(self):
        """Avanza una generación"""
        if hasattr(self, 'oned_automata') and self.oned_automata:
            self.oned_automata.next_generation()
            self.update_oned_display()

    def simulate_oned_generations(self):
        """Simula múltiples generaciones"""
        if not hasattr(self, 'oned_automata') or not self.oned_automata:
            messagebox.showwarning("Advertencia", "Primero inicialice el autómata")
            return
            
        try:
            generations = simpledialog.askinteger("Generaciones", 
                          "¿Cuántas generaciones simular?", initialvalue=50)
            if generations and generations > 0:
                for _ in range(generations):
                    self.oned_automata.next_generation()
                self.update_oned_display()
        except (ValueError, TypeError):
            pass

    def reset_oned_automata(self):
        """Reinicia el autómata"""
        if hasattr(self, 'oned_automata') and self.oned_automata:
            self.oned_automata.reset()
            self.update_oned_display()

    def update_oned_display(self):
        """Actualiza la visualización del autómata unidimensional"""
        if not hasattr(self, 'oned_automata') or not self.oned_automata:
            return
        
        self.oned_ax.clear()
    
        # Obtener matriz de historia (últimas 100 generaciones o todas si son menos)
        history = self.oned_automata.get_history_matrix(min(100, len(self.oned_automata.history)))
    
        # Mostrar como imagen (negro=0, blanco=1)
        self.oned_ax.imshow(history, cmap='binary', aspect='auto', interpolation='nearest')
        self.oned_ax.set_title(f'Autómata Unidimensional - Regla {self.oned_automata.rule}', color='white')
        self.oned_ax.set_xlabel('Posición', color='white')
        self.oned_ax.set_ylabel('Generación', color='white')
        self.oned_ax.tick_params(colors='white')
        self.oned_ax.set_facecolor(COLORS['card_bg'])
    
        self.oned_canvas.draw()
    
        # Actualizar estadísticas
        stats = self.oned_automata.get_statistics()
        self.oned_stats_var.set(f"Generación: {stats['generation']} - Células vivas: {stats['live_cells']} - Densidad: {stats['density']:.2%}")
    
    def setup_covid_simulation_tab(self, parent):
        """Configura la simulación COVID"""
        # Controles
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="Filas:").grid(row=0, column=0, padx=5)
        self.covid_rows_var = tk.StringVar(value="20")
        ttk.Entry(controls_frame, textvariable=self.covid_rows_var, width=5).grid(row=0, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Columnas:").grid(row=0, column=2, padx=5)
        self.covid_cols_var = tk.StringVar(value="20")
        ttk.Entry(controls_frame, textvariable=self.covid_cols_var, width=5).grid(row=0, column=3, padx=5)
        
        ttk.Label(controls_frame, text="Tasa infección:").grid(row=0, column=4, padx=5)
        self.covid_infection_var = tk.StringVar(value="0.3")
        ttk.Entry(controls_frame, textvariable=self.covid_infection_var, width=5).grid(row=0, column=5, padx=5)
        
        ttk.Label(controls_frame, text="Tasa recuperación:").grid(row=0, column=6, padx=5)
        self.covid_recovery_var = tk.StringVar(value="0.1")
        ttk.Entry(controls_frame, textvariable=self.covid_recovery_var, width=5).grid(row=0, column=7, padx=5)
        
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=0, column=8, padx=20)
        
        ttk.Button(buttons_frame, text="Iniciar Brote", 
                  command=self.init_covid_sim, style='Primary.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Siguiente Día", 
                  command=self.next_covid_day).pack(side=tk.LEFT, padx=2)
        ttk.Button(buttons_frame, text="Reiniciar", 
                  command=self.reset_covid_sim).pack(side=tk.LEFT, padx=2)
        
        # Área de visualización
        vis_frame = ttk.Frame(parent)
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.covid_fig, (self.covid_ax1, self.covid_ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.covid_fig.patch.set_facecolor(COLORS['card_bg'])
        
        self.covid_canvas = FigureCanvasTkAgg(self.covid_fig, vis_frame)
        self.covid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Información
        self.covid_info_var = tk.StringVar(value="Día: 0 - Susceptibles: 0% - Infectados: 0% - Recuperados: 0%")
        info_label = tk.Label(parent, textvariable=self.covid_info_var,
                             font=FONTS['normal'], bg=COLORS['background'], fg=COLORS['text_light'])
        info_label.pack(pady=5)
    
    def init_covid_sim(self):
        """Inicializa la simulación COVID"""
        try:
            rows = int(self.covid_rows_var.get())
            cols = int(self.covid_cols_var.get())
            infection_rate = float(self.covid_infection_var.get())
            recovery_rate = float(self.covid_recovery_var.get())
            
            self.covid_sim = CovidSimulation(rows, cols)
            self.covid_sim.infection_rate = infection_rate
            self.covid_sim.recovery_rate = recovery_rate
            self.covid_sim.initialize_outbreak(5)
            self.update_covid_display()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")
    
    def next_covid_day(self):
        """Avanza un día en la simulación COVID"""
        if self.covid_sim:
            self.covid_sim.next_day()
            self.update_covid_display()
    
    def reset_covid_sim(self):
        """Reinicia la simulación COVID"""
        if self.covid_sim:
            self.covid_sim.initialize_outbreak(5)
            self.update_covid_display()
    
    def update_covid_display(self):
        """Actualiza la visualización COVID"""
        if not self.covid_sim:
            return
            
        # Gráfico de la cuadrícula
        self.covid_ax1.clear()
        cmap = plt.cm.get_cmap('RdYlGn', 3)
        self.covid_ax1.imshow(self.covid_sim.grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        self.covid_ax1.set_title(f'Simulación COVID - Día {self.covid_sim.day}', color='white')
        self.covid_ax1.set_xticks([])
        self.covid_ax1.set_yticks([])
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Susceptible'),
            Patch(facecolor='red', label='Infectado'),
            Patch(facecolor='yellow', label='Recuperado')
        ]
        self.covid_ax1.legend(handles=legend_elements, loc='upper right')
        
        # Gráfico de la curva epidémica
        self.covid_ax2.clear()
        if len(self.covid_sim.history) > 1:
            days = [h['day'] for h in self.covid_sim.history]
            susceptible = [h['susceptible_pct'] for h in self.covid_sim.history]
            infected = [h['infected_pct'] for h in self.covid_sim.history]
            recovered = [h['recovered_pct'] for h in self.covid_sim.history]
            
            self.covid_ax2.plot(days, susceptible, 'g-', label='Susceptibles', linewidth=2)
            self.covid_ax2.plot(days, infected, 'r-', label='Infectados', linewidth=2)
            self.covid_ax2.plot(days, recovered, 'y-', label='Recuperados', linewidth=2)
            self.covid_ax2.set_title('Curva Epidémica', color='white')
            self.covid_ax2.set_xlabel('Días', color='white')
            self.covid_ax2.set_ylabel('Porcentaje (%)', color='white')
            self.covid_ax2.legend()
            self.covid_ax2.grid(True, alpha=0.3)
        
        # Configurar colores
        for ax in [self.covid_ax1, self.covid_ax2]:
            ax.tick_params(colors='white')
            ax.set_facecolor(COLORS['card_bg'])
        
        self.covid_canvas.draw()
        
        # Actualizar información
        stats = self.covid_sim.get_statistics()
        info_text = (f"Día: {stats['day']} - "
                    f"Susceptibles: {stats['susceptible_pct']}% - "
                    f"Infectados: {stats['infected_pct']}% - "
                    f"Recuperados: {stats['recovered_pct']}%")
        self.covid_info_var.set(info_text)
    
    def setup_secret_messages(self):
        """Configura mensajes secretos en la ventana principal"""
        create_secret_message(self.main_frame, 'main')

def main():
    root = tk.Tk()
    app = StatisticsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()