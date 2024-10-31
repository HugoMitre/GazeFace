import os  # Librería para trabajar con archivos y directorios
import radon.complexity as cc  # Librería para calcular la complejidad ciclomática
import subprocess  # Permite ejecutar comandos del sistema
import matplotlib.pyplot as plt  # Librería para crear gráficos
import numpy as np  # Librería para cálculos numéricos
from fpdf import FPDF  # Librería para generar PDFs

# Ruta de tu proyecto. Cambia esta ruta a la de tu proyecto en caso de ser diferente.
project_path = '/Users/hmitre/Library/CloudStorage/GoogleDrive-hmitre@cimat.mx/Mi unidad/CIMAT_OFICIAL/CODING/GazeFace'

# Función para calcular la complejidad ciclomática de cada archivo Python en el proyecto
def get_complexity_metrics(path):
    metrics = []  # Lista donde se almacenará la complejidad de cada función/clase encontrada
    for root, _, files in os.walk(path):  # Recorre todos los archivos del proyecto
        for file in files:
            if file.endswith('.py'):  # Solo analiza archivos .py
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    source = f.read()  # Lee el contenido del archivo
                    blocks = cc.cc_visit(source)  # Calcula la complejidad ciclomática
                    for block in blocks:
                        metrics.append(block.complexity)  # Almacena la complejidad de cada bloque
    return metrics

# Función para contar el total de líneas de código (LoC) y líneas de comentarios en el proyecto
def get_size_metrics(path):
    locs, comments = 0, 0  # Variables para líneas de código y comentarios
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    source = f.readlines()  # Lee todas las líneas del archivo
                    locs += len(source)  # Cuenta todas las líneas
                    comments += sum(1 for line in source if line.strip().startswith('#'))  # Cuenta líneas que empiezan con #
    return locs, comments

# Función para obtener el índice de mantenibilidad del código usando pylint
def get_quality_metrics(path):
    # Ejecuta pylint en todos los archivos .py del proyecto
    result = subprocess.run(['pylint', path + '/*.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pylint_output = result.stdout.splitlines()  # Divide la salida en líneas
    # Extrae el índice de mantenibilidad de la salida de pylint
    maintainability_score = [float(line.split()[6]) for line in pylint_output if line.startswith("Your code has been rated at")]
    return maintainability_score[0] if maintainability_score else 0  # Devuelve el índice o 0 si no hay resultados

# Calcula las métricas del proyecto
complexity_scores = get_complexity_metrics(project_path)  # Complejidad ciclomática
total_loc, total_comments = get_size_metrics(project_path)  # Líneas de código y comentarios
maintainability_index = get_quality_metrics(project_path)  # Índice de mantenibilidad

# Calcula la complejidad promedio y la densidad de comentarios
avg_complexity = np.mean(complexity_scores)  # Complejidad promedio del proyecto
comment_density = (total_comments / total_loc) * 100 if total_loc else 0  # Porcentaje de comentarios

# Función para crear y guardar gráficos como imágenes
def save_plot(metric_name, value, description, filename):
    # Crea un gráfico de barras para la métrica con título y descripción
    plt.figure(figsize=(6, 4))  # Tamaño del gráfico
    plt.bar([metric_name], [value], color='skyblue')  # Genera una barra con el nombre y valor de la métrica
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.title(metric_name)
    plt.legend([description])  # Agrega una leyenda que describe la métrica
    plt.savefig(filename)  # Guarda el gráfico como archivo .png
    plt.close()  # Cierra el gráfico

# Crear gráficos individuales con descripción y guardar cada uno como imagen
save_plot('Complejidad Promedio', avg_complexity, 
          'La complejidad promedio refleja la dificultad del código.\nValores altos indican mayor complejidad.', 
          'complejidad_promedio.png')

save_plot('Líneas de Código', total_loc, 
          'Cantidad total de líneas de código.\nUn proyecto más grande tendrá un valor más alto.', 
          'lineas_codigo.png')

save_plot('Densidad de Comentarios', comment_density, 
          'Porcentaje de comentarios sobre el total de líneas de código.\nMayor densidad indica más documentación.', 
          'densidad_comentarios.png')

save_plot('Índice de Mantenibilidad', maintainability_index, 
          'Indica qué tan fácil es mantener el código.\nValores altos son mejores para mantenibilidad.', 
          'indice_mantenibilidad.png')

# Crear el archivo PDF y agregar cada gráfico con su descripción
pdf = FPDF()  # Crea una instancia del PDF
pdf.add_page()  # Agrega una página nueva al PDF
pdf.set_auto_page_break(auto=True, margin=15)  # Configura márgenes y salto automático de página
pdf.set_font("Arial", "B", 16)  # Configura la fuente del título

# Agrega Complejidad Promedio al PDF
pdf.cell(200, 10, "Complejidad Promedio del Código", ln=True, align="C")  # Título centrado
pdf.image('complejidad_promedio.png', x=60, w=90)  # Inserta el gráfico en el PDF
pdf.ln(10)  # Salto de línea
pdf.set_font("Arial", "", 12)  # Cambia el tamaño de fuente para el texto
pdf.multi_cell(0, 10, 
               "La complejidad ciclomática promedio mide la dificultad del código.\n"
               "Valores altos indican más caminos de ejecución y complejidad mayor, lo que podría\n"
               "hacer el código difícil de mantener. Valores bajos indican una estructura simple\n"
               "y directa, que generalmente es más fácil de mantener.\n\n")

# Agrega Líneas de Código al PDF
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Líneas de Código", ln=True, align="C")
pdf.image('lineas_codigo.png', x=60, w=90)
pdf.ln(10)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, 
               "Este valor indica el tamaño del proyecto en términos de líneas de código (LoC).\n"
               "Un número elevado puede significar que el proyecto es grande o que no está\n"
               "suficientemente modularizado. Menor cantidad de líneas puede ser una señal\n"
               "de código eficiente o de un proyecto más pequeño.\n\n")

# Agrega Densidad de Comentarios al PDF
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Densidad de Comentarios", ln=True, align="C")
pdf.image('densidad_comentarios.png', x=60, w=90)
pdf.ln(10)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, 
               "La densidad de comentarios es el porcentaje de líneas de comentarios en\n"
               "relación con el total de líneas de código. Una alta densidad de comentarios\n"
               "indica una mejor documentación, lo que facilita el mantenimiento y la colaboración.\n\n")

# Agrega Índice de Mantenibilidad al PDF
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Índice de Mantenibilidad", ln=True, align="C")
pdf.image('indice_mantenibilidad.png', x=60, w=90)
pdf.ln(10)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, 
               "El índice de mantenibilidad mide qué tan fácil será mantener el código a lo largo\n"
               "del tiempo. Un índice alto indica un código bien estructurado y documentado, lo que\n"
               "facilita futuras modificaciones y ajustes.\n\n")

# Guardar el PDF en el directorio actual
pdf.output("Reporte_Metricas_Codigo.pdf")

print("PDF creado exitosamente como 'Reporte_Metricas_Codigo.pdf'")