#!/usr/bin/env python3
"""
SISTEMA DE SIMULACIÓN ESTADÍSTICA
Punto de entrada principal de la aplicación
"""

import tkinter as tk
import sys
import os

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importación directa
from app import StatisticsApp

def main():
    """Función principal que inicia la aplicación"""
    try:
        print("🚀 Iniciando Sistema de Simulación Estadística...")
        root = tk.Tk()
        app = StatisticsApp(root)
        root.mainloop()
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")
        import traceback
        traceback.print_exc()
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()