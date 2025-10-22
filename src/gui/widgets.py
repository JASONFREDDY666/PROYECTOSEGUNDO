"""
Widgets personalizados y componentes reutilizables
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

def create_secret_message(parent, tab_key):
    """Crea un mensaje secreto oculto en cada pestaña"""
    
    def show_secret_message():
        """Muestra el mensaje secreto"""
        message = config.SECRET_MESSAGES[tab_key]
        
        # Crear ventana emergente
        popup = tk.Toplevel(parent)
        popup.title("💫 Mensaje Secreto")
        popup.geometry("400x150")
        popup.configure(bg=config.COLORS['card_bg'])
        popup.resizable(False, False)
        
        # Centrar ventana
        popup.transient(parent)
        popup.grab_set()
        
        # Mensaje
        msg_label = tk.Label(
            popup,
            text=message,
            font=config.FONTS['heading'],
            bg=config.COLORS['card_bg'],
            fg=config.COLORS['accent'],
            wraplength=350,
            justify=tk.CENTER
        )
        msg_label.pack(expand=True, padx=20, pady=20)
        
        # Botón de cierre
        close_btn = ttk.Button(
            popup,
            text="✨ Cerrar",
            command=popup.destroy,
            style='Primary.TButton'
        )
        close_btn.pack(pady=10)
        
    # Crear botón secreto (invisible hasta pasar el mouse)
    secret_btn = ttk.Button(
        parent,
        text="🔍",
        command=show_secret_message,
        width=3
    )
    
    # Posicionar en esquina inferior derecha
    secret_btn.place(relx=0.98, rely=0.98, anchor='se')
    
    # Configurar tooltip
    def on_enter(e):
        secret_btn.configure(text="💫")
    
    def on_leave(e):
        secret_btn.configure(text="🔍")
    
    secret_btn.bind("<Enter>", on_enter)
    secret_btn.bind("<Leave>", on_leave)
    
    return secret_btn