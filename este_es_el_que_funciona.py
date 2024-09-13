import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Crear el DataFrame con los datos proporcionados
data = {
    'Componente': [1, 2, 1, 2, 1, 2, 1, 2],
    'sexo': [-0.509, 0.925, 0.755, 0.251, 0.033, 0.200, -0.887, 0.799],
    'divertid': [-0.028, -0.186, -0.389, 0.815, 0.873, 0.825, -0.029, 0.034]
}

df = pd.DataFrame(data)

# Mostrar el DataFrame
print("DataFrame con los datos proporcionados:")
print(df)

# Ejemplo de manipulación de datos: Seleccionar una columna
print("\nColumna 'sexo':")
print(df['sexo'])

# Ejemplo de manipulación de datos: Filtrar filas donde 'sexo' es mayor que 0
filtered_df = df[df['sexo'] > 0]
print("\nFiltrado donde 'sexo' es mayor que 0:")
print(filtered_df)

# Crear gráficos
# Gráfico 1: Dispersión de 'sexo' vs 'divertid'
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(df['sexo'], df['divertid'], c='blue', label='Datos')
plt.xlabel('Sexo')
plt.ylabel('Divertid')
plt.title('Gráfico de Dispersión de Sexo vs Divertid')
plt.legend()
plt.grid(True)

# Configuración y ajuste de KMeans con 4 clusters
X = df[['sexo', 'divertid']]

k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

# Obtener etiquetas y centros de clusters
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

# Gráfico 2: Clusters con 4 grupos
plt.subplot(1, 3, 2)
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

for k, col in zip(range(len(k_means_cluster_centers)), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    plt.scatter(X[my_members]['sexo'], X[my_members]['divertid'], c=col, label=f'Cluster {k}')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

plt.xlabel('Sexo')
plt.ylabel('Divertid')
plt.title('KMeans con 4 Clusters')
plt.legend()
plt.grid(True)

# Configuración y ajuste de KMeans con 3 clusters
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)

# Gráfico 3: Clusters con 3 grupos
plt.subplot(1, 3, 3)
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))

for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    plt.scatter(X[my_members]['sexo'], X[my_members]['divertid'], c=col, label=f'Cluster {k}')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

plt.xlabel('Sexo')
plt.ylabel('Divertid')
plt.title('KMeans con 3 Clusters')
plt.legend()
plt.grid(True)

# Mostrar todos los gráficos
plt.tight_layout()
plt.show()
