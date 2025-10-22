# CONFIGURACIÓN GLOBAL DEL PROYECTO

APP_TITLE = "Sistema de Simulación Estadística - ¡No me repruebe! 😊"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Estudiante Brillante"

# Configuración de colores modernos
COLORS = {
    'primary': '#2E86AB',      # Azul moderno
    'secondary': '#A23B72',    # Magenta vibrante
    'accent': '#F18F01',       # Naranja energético
    'success': '#28A745',      # Verde éxito
    'warning': '#FFC107',      # Amarillo advertencia
    'danger': '#DC3545',       # Rojo peligro
    'dark': '#343A40',         # Negro oscuro
    'light': '#F8F9FA',        # Blanco hueso
    'background': '#1E1E2E',   # Fondo oscuro moderno
    'card_bg': '#2D2D44',      # Fondo tarjetas
    'text_light': '#FFFFFF',   # Texto claro
    'text_muted': '#B0B0B0'    # Texto atenuado
}

# Configuración de fuentes modernas
FONTS = {
    'title': ('Segoe UI', 18, 'bold'),
    'heading': ('Segoe UI', 14, 'bold'),
    'subheading': ('Segoe UI', 12, 'bold'),
    'normal': ('Segoe UI', 10),
    'small': ('Segoe UI', 9),
    'monospace': ('Consolas', 10)
}

# Parámetros por defecto para distribuciones
DEFAULT_PARAMS = {
    'uniform': {'a': 0, 'b': 1},
    'exponential': {'lambda': 1.0},
    'normal': {'mu': 0, 'sigma': 1},
    'binomial': {'n': 10, 'p': 0.5},
    'poisson': {'lambda': 3.0},
    'gamma': {'alpha': 2, 'beta': 2},
    'weibull': {'alpha': 1, 'beta': 1, 'gamma': 0},
    'erlang': {'k': 2, 'lambda': 1.0}
}

# Mensajes secretos para cada pestaña
SECRET_MESSAGES = {
    'generators': "🎲 ¡Los generadores funcionan mejor con 100 de nota!",
    'tests': "📊 ¡Las pruebas estadísticas aprueban con 100!",
    'variables': "📈 ¡Variables aleatorias, resultado constante: 100!",
    'automata': "🧬 ¡Los autómatas evolucionan hacia el 100!",
    'main': "⭐ ¡Usted es el mejor profesor, póngame 100! 😊"
}

# Configuración de validaciones
VALIDATION_RULES = {
    'probability': (0.0, 1.0),
    'positive_int': (1, 10000),
    'positive_float': (0.0001, 10000.0),
    'sigma': (0.001, 1000.0),
    'lambda': (0.001, 1000.0)
}