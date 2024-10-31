import git  # Importa la biblioteca GitPython para trabajar con el repositorio Git
import matplotlib.pyplot as plt  # Importa matplotlib para graficar los resultados

# Configura la ruta del repositorio Git en el sistema local. Cambia esta ruta a la ubicación de tu repositorio.
repo_path = '/Users/hmitre/Library/CloudStorage/GoogleDrive-hmitre@cimat.mx/Mi unidad/CIMAT_OFICIAL/CODING/GazeFace'
repo = git.Repo(repo_path)  # Carga el repositorio en la variable 'repo' usando GitPython

# Diccionario para almacenar el balance de líneas de código (inserciones menos eliminaciones) para cada rama
branch_data = {}

# Itera sobre todas las ramas locales del repositorio
for branch in repo.branches:
    repo.git.checkout(branch)  # Cambia a la rama actual para poder analizarla
    commits = list(repo.iter_commits(branch))  # Obtiene la lista de todos los commits en la rama actual

    # Verifica si la rama tiene al menos dos commits (necesarios para calcular la diferencia de código)
    if len(commits) > 1:
        # Ejecuta el comando `git diff` para obtener un resumen de cambios en el último commit
        # `--shortstat` resume el total de inserciones y eliminaciones en una línea de salida
        diff = repo.git.diff('HEAD~1', '--shortstat')
        
        # Calcula el total de inserciones contando las líneas que contienen 'insertion'
        insertions = sum(int(line.split()[0]) for line in diff.splitlines() if 'insertion' in line)
        # Calcula el total de eliminaciones contando las líneas que contienen 'deletion'
        deletions = sum(int(line.split()[0]) for line in diff.splitlines() if 'deletion' in line)
        
        # Guarda el balance (inserciones - eliminaciones) en el diccionario `branch_data` usando el nombre de la rama
        branch_data[branch.name] = insertions - deletions
    else:
        # Imprime un mensaje si la rama no tiene suficientes commits para calcular diferencias
        print(f"La rama {branch.name} no tiene suficientes commits y será omitida.")

# Configuración de la gráfica para mostrar los resultados
plt.figure(figsize=(10, 6))  # Define el tamaño de la figura de la gráfica en pulgadas
plt.bar(branch_data.keys(), branch_data.values(), color='skyblue')  # Crea una gráfica de barras con nombres de ramas y su balance de líneas
plt.xlabel('Ramas')  # Etiqueta del eje X, mostrando los nombres de las ramas
plt.ylabel('Líneas de Código')  # Etiqueta del eje Y, representando el balance de líneas de código
plt.title('Líneas de Código por Rama en el Proyecto')  # Título de la gráfica
plt.xticks(rotation=45)  # Rota los nombres de las ramas en el eje X para facilitar la lectura
plt.tight_layout()  # Ajusta automáticamente los márgenes de la gráfica para evitar superposiciones
plt.show()  # Muestra la gráfica en la pantalla
