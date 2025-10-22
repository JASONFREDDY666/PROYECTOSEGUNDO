"""
Funciones para configuración y personalización de gráficos
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def configure_matplotlib_style():
    """
    Configura el estilo de matplotlib para coincidir con el tema de la aplicación
    """
    plt.rcParams.update({
        'figure.facecolor': '#1E1E2E',
        'axes.facecolor': '#2D2D44',
        'axes.edgecolor': '#555555',
        'axes.labelcolor': '#FFFFFF',
        'axes.titlecolor': '#FFFFFF',
        'text.color': '#FFFFFF',
        'xtick.color': '#FFFFFF',
        'ytick.color': '#FFFFFF',
        'grid.color': '#404040',
        'grid.alpha': 0.3,
        'figure.titlesize': 14,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'font.family': 'Segoe UI'
    })

def create_custom_colormap(name='custom_blue', colors=None):
    """
    Crea un mapa de colores personalizado
    """
    if colors is None:
        colors = ['#1E1E2E', '#2E86AB', '#F18F01']  # Dark blue to orange
    
    return LinearSegmentedColormap.from_list(name, colors)

def setup_automata_plot(ax, title, xlabel='Posición', ylabel='Generación'):
    """
    Configura un gráfico para autómatas celulares
    """
    ax.clear()
    ax.set_title(title, color='white', fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='#404040')
    ax.set_facecolor('#2D2D44')

def setup_distribution_plot(ax, title, xlabel='Valor', ylabel='Densidad/Frecuencia'):
    """
    Configura un gráfico para distribuciones
    """
    ax.clear()
    ax.set_title(title, color='white', fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='#404040')
    ax.set_facecolor('#2D2D44')

def create_comparison_plot(fig, axes, data_sets, labels, plot_type='line'):
    """
    Crea un gráfico de comparación entre múltiples conjuntos de datos
    """
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#FFC107']
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        color = colors[i % len(colors)]
        
        if plot_type == 'line':
            axes.plot(data, label=label, color=color, linewidth=2, alpha=0.8)
        elif plot_type == 'bar':
            x_pos = np.arange(len(data)) + i * 0.2
            axes.bar(x_pos, data, width=0.2, label=label, color=color, alpha=0.8)
        elif plot_type == 'hist':
            axes.hist(data, bins=30, alpha=0.6, label=label, color=color, density=True)
    
    axes.legend(facecolor='#2D2D44', edgecolor='#555555', labelcolor='white')
    setup_distribution_plot(axes, 'Comparación de Distribuciones')

def save_plot_as_image(fig, filename, dpi=150, transparent=False):
    """
    Guarda una figura matplotlib como imagen
    """
    try:
        fig.savefig(
            filename,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            transparent=transparent,
            bbox_inches='tight',
            pad_inches=0.1
        )
        return True
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return False

def create_3d_surface_plot(fig, data, title="Superficie 3D"):
    """
    Crea un gráfico de superficie 3D
    """
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, data, cmap='viridis', alpha=0.8)
    
    ax.set_title(title, color='white')
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    
    # Configurar colores de los ejes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    return ax, surf