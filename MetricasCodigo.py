import os
import radon.complexity as cc
import radon.metrics as rm
from pylint import epylint as lint
import matplotlib.pyplot as plt
import numpy as np

# Ruta del proyecto: especifica aquí la ruta del proyecto de Python que deseas analizar
project_path = 'ruta/del/proyecto'

# Función para obtener métricas de complejidad con radon
# get_complexity_metrics(path): Recorre cada archivo .py en la carpeta del proyecto para calcular 
# la complejidad ciclomática. Utiliza radon para analizar la complejidad de funciones y clases y
#  guarda cada valor en una lista.
def get_complexity_metrics(path):
    """Analiza la complejidad ciclomática de cada archivo .py en el proyecto."""
    metrics = []
    # Recorre cada archivo en la carpeta del proyecto
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):  # Solo analizamos archivos .py
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    source = f.read()
                    # Calcula la complejidad ciclomática para cada función/clase
                    blocks = cc.cc_visit(source)
                    for block in blocks:
                        metrics.append(block.complexity)  # Guarda la complejidad de cada bloque
    return metrics

# Función para obtener número de líneas de código y comentarios
# get_size_metrics(path): Calcula el número de líneas de código (LoC) y 
# la densidad de comentarios. Lee cada archivo .py, cuenta todas las líneas 
# y separa las que son comentarios (líneas que comienzan con #).
def get_size_metrics(path):
    """Cuenta el número total de líneas de código (LoC) y líneas de comentarios."""
    locs, comments = 0, 0  # Inicializa contadores
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    source = f.readlines()  # Lee todas las líneas del archivo
                    locs += len(source)  # Cuenta las líneas de código
                    # Cuenta las líneas de comentarios
                    comments += sum(1 for line in source if line.strip().startswith('#'))
    return locs, comments

# Función para obtener índice de mantenibilidad y calidad
# get_quality_metrics(path): Ejecuta pylint en el proyecto y obtiene el índice de mantenibilidad. 
# Este índice evalúa qué tan fácil será mantener el código en el tiempo.
def get_quality_metrics(path):
    """Usa pylint para obtener el índice de mantenibilidad del proyecto."""
    pylint_output = []  # Aquí se guardarán los resultados de pylint
    lint.py_run(path + '/*.py', return_std=True, script='pylint')
    for line in lint.stdout:
        pylint_output.append(line)
    
    # Busca el índice de mantenibilidad en la salida de pylint
    maintainability_score = [float(line.split()[2]) for line in pylint_output if line.startswith("Your code has been rated at")]
    return maintainability_score[0] if maintainability_score else 0

# Obtener datos para cada métrica usando las funciones anteriores
complexity_scores = get_complexity_metrics(project_path)
total_loc, total_comments = get_size_metrics(project_path)
maintainability_index = get_quality_metrics(project_path)

# Cálculo de métricas adicionales
# avg_complexity: Calcula la complejidad promedio a partir de la lista de complejidades obtenidas.
avg_complexity = np.mean(complexity_scores)  # Calcula la complejidad promedio

# comment_density: Calcula el porcentaje de líneas de comentarios respecto al total de líneas de código.
comment_density = (total_comments / total_loc) * 100 if total_loc else 0  # Calcula la densidad de comentarios

# Preparación de los datos para graficar
metricas = ['Complejidad Promedio', 'Líneas de Código', 'Densidad de Comentarios', 'Índice de Mantenibilidad']
valores = [avg_complexity, total_loc, comment_density, maintainability_index]

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.bar(metricas, valores, color=['skyblue', 'salmon', 'gold', 'lightgreen'])
plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.title('Resumen de Métricas del Código')
plt.show()
