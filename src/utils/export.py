"""
Funciones para exportación de datos y resultados
"""

import pandas as pd
import json
import csv
import os
from datetime import datetime
from tkinter import filedialog, messagebox

def export_data_to_csv(data, filename=None, metadata=None):
    """
    Exporta datos a un archivo CSV
    """
    try:
        if filename is None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not filename:
                return False
        
        df = pd.DataFrame(data)
        
        # Agregar metadatos como comentarios
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
            df.to_csv(f, index=False)
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al exportar CSV: {e}")
        return False

def export_data_to_json(data, filename=None, metadata=None):
    """
    Exporta datos a un archivo JSON
    """
    try:
        if filename is None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return False
        
        export_data = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al exportar JSON: {e}")
        return False

def generate_report(results, report_type="comprehensive"):
    """
    Genera un reporte completo de los resultados
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'report_type': report_type,
        'summary': {},
        'detailed_results': results
    }
    
    # Generar resumen basado en el tipo de reporte
    if report_type == "statistical_tests":
        report['summary'] = _generate_statistical_summary(results)
    elif report_type == "distribution_analysis":
        report['summary'] = _generate_distribution_summary(results)
    elif report_type == "automata_simulation":
        report['summary'] = _generate_automata_summary(results)
    
    return report

def _generate_statistical_summary(results):
    """Genera resumen para pruebas estadísticas"""
    summary = {}
    
    if 'chi_square' in results:
        chi_result = results['chi_square']
        summary['uniformity'] = {
            'test': 'Chi-cuadrado',
            'statistic': chi_result.get('chi_square_statistic', 'N/A'),
            'is_uniform': chi_result.get('is_uniform', False),
            'result': 'APROBADO' if chi_result.get('is_uniform', False) else 'RECHAZADO'
        }
    
    if 'runs_test' in results:
        runs_result = results['runs_test']
        summary['independence'] = {
            'test': 'Prueba de Rachas',
            'statistic': runs_result.get('z_statistic', 'N/A'),
            'is_independent': runs_result.get('is_independent', False),
            'result': 'APROBADO' if runs_result.get('is_independent', False) else 'RECHAZADO'
        }
    
    return summary

def _generate_distribution_summary(results):
    """Genera resumen para análisis de distribuciones"""
    summary = {
        'distribution_type': results.get('distribution_type', 'Desconocida'),
        'sample_size': results.get('sample_size', 0),
        'parameters': results.get('parameters', {})
    }
    
    # Agregar estadísticas básicas si están disponibles
    if 'statistics' in results:
        stats = results['statistics']
        summary['key_statistics'] = {
            'mean': stats.get('mean', 'N/A'),
            'std_dev': stats.get('std', 'N/A'),
            'variance': stats.get('variance', 'N/A'),
            'range': stats.get('range', 'N/A')
        }
    
    return summary

def _generate_automata_summary(results):
    """Genera resumen para simulaciones de autómatas"""
    summary = {
        'automata_type': results.get('automata_type', 'Desconocido'),
        'generations': results.get('total_generations', 0),
        'initial_conditions': results.get('initial_conditions', {}),
        'final_state': results.get('final_state', {})
    }
    
    return summary

def save_simulation_state(simulation, filename=None):
    """
    Guarda el estado actual de una simulación
    """
    try:
        if filename is None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return False
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'simulation_type': type(simulation).__name__,
            'state': simulation.__dict__ if hasattr(simulation, '__dict__') else str(simulation)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al guardar estado: {e}")
        return False

def load_simulation_state(filename=None, simulation_class=None):
    """
    Carga el estado de una simulación desde archivo
    """
    try:
        if filename is None:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Aquí podrías implementar la lógica para reconstruir la simulación
        # basándote en simulation_class y los datos del estado
        
        return state
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar estado: {e}")
        return None

def export_plot_data(fig, filename=None):
    """
    Exporta los datos de un gráfico matplotlib
    """
    try:
        if filename is None:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return False
        
        plot_data = {}
        
        for i, ax in enumerate(fig.get_axes()):
            axis_data = {}
            
            # Obtener datos de líneas
            lines = ax.get_lines()
            for j, line in enumerate(lines):
                axis_data[f'line_{j}'] = {
                    'xdata': line.get_xdata().tolist(),
                    'ydata': line.get_ydata().tolist(),
                    'label': line.get_label()
                }
            
            # Obtener datos de barras
            patches = ax.patches
            if patches:
                bar_data = []
                for patch in patches:
                    bar_data.append({
                        'x': patch.get_x(),
                        'y': patch.get_height(),
                        'width': patch.get_width()
                    })
                axis_data['bars'] = bar_data
            
            plot_data[f'axis_{i}'] = axis_data
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, indent=2)
        
        return True
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al exportar datos del gráfico: {e}")
        return False