Sistema de Simulación Estadística

Aplicación de escritorio desarrollada en Python, orientada a la generación, análisis y simulación estadística de datos mediante generadores pseudoaleatorios, pruebas estadísticas, variables aleatorias y autómatas celulares.
El sistema está diseñado con un enfoque académico y experimental, integrando módulos visuales y cálculos automatizados para validación estadística.

Requisitos e Instalación
Requisitos del Sistema

Python 3.8 o superior

Sistema operativo: Windows, Linux o macOS

Memoria RAM: mínimo 4 GB (recomendado 8 GB)

Espacio en disco: 100 MB libres

Dependencias necesarias
pip install numpy matplotlib scipy pandas tkinter

Pasos de instalación

Descargar el proyecto:

git clone https://github.com/JASONFREDDY666/PROYECTOSEGUNDO.git
cd sistema-simulacion-estadistica


Verificar la estructura de archivos:

app.py
distributions.py
validators.py
chi_square.py
utils/
  ├── helpers.py
  ├── plotting.py
  └── export.py
automata/
  └── one_dimensional.py


Ejecutar la aplicación:

python app.py

Solución de Problemas Comunes
Problema	Solución
Error de módulos	Asegurarse de que todos los archivos .py se encuentren en la misma carpeta.
Error gráfico en Linux	Ejecutar: export MPLBACKEND=Agg
Tkinter no encontrado	Instalar python3-tk (Linux) o reinstalar Python con soporte para Tkinter (Windows).
Descripción de la Interfaz
Pestaña: Generadores
<img width="1193" height="813" alt="image" src="https://github.com/user-attachments/assets/b1bf2b0a-4529-4699-bce1-bf8551b75d32" />

Incluye un selector de generadores pseudoaleatorios (Congruencial, Cuadrados Medios y Fibonacci).
Permite definir semilla, cantidad de números y visualizar los resultados en un histograma.
Incluye opciones para generar, probar calidad y exportar los datos obtenidos.

Pestaña: Variables Aleatorias
<img width="1170" height="795" alt="image" src="https://github.com/user-attachments/assets/62d2d45d-c0ed-49ce-98de-20cefb40c3a1" />

Incluye distribuciones continuas (Uniforme, K-Erlang, Exponencial, Normal, Gamma y Weibull) y discretas (Uniforme, Bernoulli, Binomial y Poisson).
Cada distribución cuenta con sus parámetros configurables y una representación gráfica mediante histogramas.

Pestaña: Pruebas Estadísticas
<img width="1159" height="792" alt="image" src="https://github.com/user-attachments/assets/c894f8e2-c0bb-4b4d-b52d-bd723159c5cf" />

Implementa las pruebas de Chi-cuadrado, Rachas y Suite completa.
Muestra resultados numéricos, valores p y conclusiones estadísticas.
Incluye un área de texto para reportes detallados.

Pestaña: Autómatas Celulares
<img width="1158" height="778" alt="image" src="https://github.com/user-attachments/assets/a4f07973-4fba-4014-a6d8-9b0c76febfc9" />

Incluye el Juego de la Vida de Conway, autómatas unidimensionales (reglas de Wolfram 0–255) y simulaciones epidemiológicas tipo SIR (COVID-19).
Permite ajustar parámetros de simulación y visualizar la cuadrícula en tiempo real.
Estado Actual del Proyecto
Estado	Descripción
Completado	Interfaz gráfica con cuatro pestañas principales: Generadores, Variables, Pruebas y Autómatas.
Generadores	Tres métodos implementados: Congruencial, Cuadrados Medios y Fibonacci.
Distribuciones	Diez distribuciones de probabilidad (seis continuas y cuatro discretas).
Pruebas estadísticas	Chi-cuadrado, Rachas y Suite completa.
Autómatas celulares	Juego de la Vida, Reglas unidimensionales y simulación epidemiológica.
Funcionalidades adicionales	Sistema de validación de parámetros, exportación de datos y generación de reportes.
Estado de Funcionamiento

El sistema permite:

Generación y análisis de datos en tiempo real.

Visualización interactiva mediante gráficos con Matplotlib.

Manejo robusto de errores y validaciones dinámicas.

Interfaz moderna, funcional y de fácil uso.

Autor: Robert Marco Copa Mamani
Proyecto: Sistema de Simulación Estadística y Autómatas Celulares
Año: 2025
