#!/usr/bin/env python3
"""
Archivo de prueba simplificado
"""

import tkinter as tk
from tkinter import ttk

# Configuración simple
COLORS = {
    'primary': '#2E86AB',
    'background': '#1E1E2E',
    'card_bg': '#2D2D44',
    'text_light': '#FFFFFF'
}

class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Simulación - Prueba")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS['background'])
        
        self.create_notebook()
        
    def create_notebook(self):
        # Crear notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear pestañas
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        tab3 = ttk.Frame(notebook)
        tab4 = ttk.Frame(notebook)
        
        notebook.add(tab1, text="🎲 Generadores")
        notebook.add(tab2, text="📊 Pruebas")
        notebook.add(tab3, text="📈 Variables")
        notebook.add(tab4, text="🧬 Autómatas")
        
        # Contenido simple
        label1 = tk.Label(tab1, text="Módulo Generadores - En desarrollo", 
                         bg=COLORS['card_bg'], fg=COLORS['text_light'])
        label1.pack(expand=True)
        
        label2 = tk.Label(tab2, text="Módulo Pruebas - En desarrollo", 
                         bg=COLORS['card_bg'], fg=COLORS['text_light'])
        label2.pack(expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()