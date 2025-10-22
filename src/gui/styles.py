"""
Módulo para estilos modernos de la interfaz
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Configurar path para importaciones
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from .. import config

def apply_modern_theme(root):
    """Aplica un tema moderno a la aplicación"""
    
    style = ttk.Style()
    
    # Configurar tema general
    style.theme_use('clam')
    
    # Configurar colores y estilos
    style.configure('TFrame', background=config.COLORS['background'])
    style.configure('TLabel', background=config.COLORS['background'], 
                   foreground=config.COLORS['text_light'], font=config.FONTS['normal'])
    style.configure('TButton', font=config.FONTS['normal'], padding=6)
    style.configure('TNotebook', background=config.COLORS['background'])
    style.configure('TNotebook.Tab', font=config.FONTS['subheading'], padding=10)
    
    # Estilo para tarjetas
    style.configure('Card.TFrame', background=config.COLORS['card_bg'], 
                   relief='raised', borderwidth=1)
    
    # Estilo para botones primarios
    style.configure('Primary.TButton', 
                   background=config.COLORS['primary'],
                   foreground=config.COLORS['text_light'],
                   focuscolor='none')
    
    # Estilo para botones de éxito
    style.configure('Success.TButton',
                   background=config.COLORS['success'],
                   foreground=config.COLORS['text_light'])
    
    # Estilo para botones de peligro
    style.configure('Danger.TButton',
                   background=config.COLORS['danger'],
                   foreground=config.COLORS['text_light'])
    
    # Estilo para entradas
    style.configure('TEntry', fieldbackground=config.COLORS['light'])
    style.configure('TCombobox', fieldbackground=config.COLORS['light'])
    
    # Configurar el fondo de la ventana principal
    root.configure(bg=config.COLORS['background'])