"""
Funciones auxiliares y de utilidad general
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import os
from datetime import datetime

# ========== CONFIGURACIÓN GLOBAL ==========

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
    'text_muted': '#B0B0B0',
    'grid_line': '#404040',
    'border': '#555555'
}

FONTS = {
    'title': ('Segoe UI', 18, 'bold'),
    'heading': ('Segoe UI', 14, 'bold'),
    'subheading': ('Segoe UI', 12, 'bold'),
    'normal': ('Segoe UI', 10),
    'small': ('Segoe UI', 9),
    'monospace': ('Consolas', 10),
    'large': ('Segoe UI', 16, 'bold')
}

SECRET_MESSAGES = {
    'generators': "🎲 ¡Los generadores funcionan mejor con 100 de nota!",
    'tests': "📊 ¡Las pruebas estadísticas aprueban con 100!",
    'variables': "📈 ¡Variables aleatorias, resultado constante: 100!",
    'automata': "🧬 ¡Los autómatas evolucionan hacia el 100!",
    'main': "⭐ ¡Usted es el mejor profesor, póngame 100! 😊"
}

def apply_modern_theme(root):
    """
    Aplica un tema moderno y oscuro a la aplicación tkinter
    """
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configurar colores de fondo
    style.configure('.', 
                   background=COLORS['background'],
                   foreground=COLORS['text_light'])
    
    # Configurar frames
    style.configure('TFrame', 
                   background=COLORS['background'],
                   relief='flat')
    
    # Configurar labels
    style.configure('TLabel',
                   background=COLORS['background'],
                   foreground=COLORS['text_light'],
                   font=FONTS['normal'])
    
    # Configurar botones
    style.configure('TButton',
                   background=COLORS['primary'],
                   foreground=COLORS['text_light'],
                   font=FONTS['normal'],
                   padding=(10, 5),
                   relief='flat',
                   borderwidth=0)
    
    style.map('TButton',
             background=[('active', COLORS['accent']),
                        ('pressed', COLORS['secondary'])])
    
    # Configurar notebook (pestañas)
    style.configure('TNotebook',
                   background=COLORS['background'],
                   borderwidth=0)
    
    style.configure('TNotebook.Tab',
                   background=COLORS['card_bg'],
                   foreground=COLORS['text_light'],
                   padding=(15, 8),
                   font=FONTS['subheading'])
    
    style.map('TNotebook.Tab',
             background=[('selected', COLORS['primary']),
                        ('active', COLORS['accent'])])
    
    # Configurar labelframes
    style.configure('TLabelframe',
                   background=COLORS['background'],
                   foreground=COLORS['text_light'],
                   borderwidth=1,
                   relief='solid')
    
    style.configure('TLabelframe.Label',
                   background=COLORS['background'],
                   foreground=COLORS['accent'],
                   font=FONTS['subheading'])
    
    # Configurar entries
    style.configure('TEntry',
                   fieldbackground=COLORS['card_bg'],
                   foreground=COLORS['text_light'],
                   borderwidth=1,
                   relief='solid')
    
    # Configurar combobox
    style.configure('TCombobox',
                   fieldbackground=COLORS['card_bg'],
                   background=COLORS['primary'],
                   foreground=COLORS['text_light'])
    
    # Configurar scrollbar
    style.configure('Vertical.TScrollbar',
                   background=COLORS['card_bg'],
                   troughcolor=COLORS['background'],
                   borderwidth=0)
    
    # Configurar la ventana principal
    root.configure(bg=COLORS['background'])

def create_secret_message(parent, tab_key):
    """
    Crea un mensaje secreto oculto en cada pestaña
    """
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

def validate_numeric_input(value, min_val=None, max_val=None):
    """
    Valida que un valor sea numérico y esté dentro de un rango
    """
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False, f"El valor debe ser mayor o igual a {min_val}"
        if max_val is not None and num > max_val:
            return False, f"El valor debe ser menor o igual a {max_val}"
        return True, num
    except ValueError:
        return False, "Por favor ingrese un valor numérico válido"

def format_statistics(stats_dict):
    """
    Formatea un diccionario de estadísticas para mostrar
    """
    formatted = "=== ESTADÍSTICAS ===\n\n"
    for key, value in stats_dict.items():
        if isinstance(value, float):
            formatted += f"{key}: {value:.4f}\n"
        else:
            formatted += f"{key}: {value}\n"
    return formatted

def generate_timestamp():
    """
    Genera un timestamp formateado para nombres de archivo
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_basic_statistics(data):
    """
    Calcula estadísticas básicas de un conjunto de datos
    """
    if not data or len(data) == 0:
        return {}
    
    data_array = np.array(data)
    
    return {
        'count': len(data),
        'mean': float(np.mean(data_array)),
        'median': float(np.median(data_array)),
        'std': float(np.std(data_array)),
        'variance': float(np.var(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'range': float(np.max(data_array) - np.min(data_array))
    }

def create_tooltip(widget, text):
    """
    Crea un tooltip para un widget
    """
    def show_tooltip(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        label = tk.Label(tooltip, text=text, background="#ffffe0", 
                        relief='solid', borderwidth=1, font=FONTS['small'])
        label.pack()
        
        def hide_tooltip():
            tooltip.destroy()
        
        widget.bind("<Leave>", lambda e: hide_tooltip())
    
    widget.bind("<Enter>", show_tooltip)