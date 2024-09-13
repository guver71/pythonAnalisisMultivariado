import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Crear el DataFrame con los datos proporcionados
data = {
    'Componente': [1, 2],
    'sexo': [-0.509, 0.925],
    'divertid': [-0.028, -0.186],
    'pidocomp': [0.755, -0.389],
    'aprendom': [0.251, 0.815],
    'excur': [0.033, 0.873],
    'quitatie': [0.200, 0.825],
    'nomeint': [-0.887, -0.029],
    'gustovis': [0.799, 0.034]
}

df = pd.DataFrame(data)

# Mostrar el DataFrame
print("DataFrame con los datos proporcionados:")
print(df)

# Extraer las características para clustering
X = df[['sexo', 'divertid']]

# Mostrar los puntos de los datos proporcionados
plt.figure(figsize=(12, 4))

# Primer gráfico: Datos proporcionados
plt.subplot(1, 3, 1)
plt.scatter(df['sexo'], df['divertid'], marker='o', color='blue')
plt.xlabel('Sexo')
plt.ylabel('Divertid')
plt.title('Datos Proporcionados')
plt.grid(True)

# Configurar y ajustar KMeans con 2 clusters
k_means = KMeans(init="k-means++", n_clusters=2, n_init=12)
k_means.fit(X)

# Obtener etiquetas y centros de clusters
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

# Segundo gráfico: Clustering con 2 clusters
plt.subplot(1, 3, 2)
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

for k, col in zip(range(len(k_means_cluster_centers)), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    plt.scatter(X[my_members]['sexo'], X[my_members]['divertid'], c=col, label=f'Cluster {k}')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

plt.xlabel('Sexo')
plt.ylabel('Divertid')
plt.title('KMeans con 2 Clusters')
plt.legend()
plt.grid(True)

# KMeans con 3 clusters (esto no se aplicará en este caso debido al número de datos, pero se incluye para fines ilustrativos)
k_means3 = KMeans(init="k-means++", n_clusters=2, n_init=12)
k_means3.fit(X)

# Tercer gráfico: Clustering con 2 clusters
plt.subplot(1, 3, 3)
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))

for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    plt.scatter(X[my_members]['sexo'], X[my_members]['divertid'], c=col, label=f'Cluster {k}')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

plt.xlabel('Sexo')
plt.ylabel('Divertid')
plt.title('KMeans con 2 Clusters')
plt.legend()
plt.grid(True)

# Mostrar todos los gráficos
plt.tight_layout()
plt.show()
