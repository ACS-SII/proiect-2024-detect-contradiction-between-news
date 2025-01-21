import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

# Încărcăm matricea de similaritate și datele originale
print("Încărcare date...")
similarity_matrix = np.load('similarity_matrix.npy')
df = pd.read_pickle('news_articles_with_embeddings.pkl')
print(f"Număr de articole încărcate: {len(df)}")

# Convertim matricea de similaritate în matrice de distanță
distance_matrix = 1 - similarity_matrix

# Funcție pentru a aplica clusterizarea și a returna rezultatele
def apply_clustering(n_clusters=None, distance_threshold=None):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                         distance_threshold=distance_threshold, 
                                         linkage='average')
    labels = clustering.fit_predict(distance_matrix)
    return labels

# Experimentăm cu diferite numere de clustere
n_clusters_list = [50, 100, 200, 500]
results = {}

for n_clusters in tqdm(n_clusters_list, desc="Aplicăm clusterizarea pentru diferite numere de clustere"):
    labels = apply_clustering(n_clusters=n_clusters)
    cluster_sizes = Counter(labels)
    results[n_clusters] = {
        'labels': labels,
        'cluster_sizes': cluster_sizes
    }
    print(f"\nNumăr de clustere: {n_clusters}")
    print(f"Distribuția mărimii clusterelor:")
    print(f"  Min: {min(cluster_sizes.values())}")
    print(f"  Max: {max(cluster_sizes.values())}")
    print(f"  Medie: {sum(cluster_sizes.values()) / len(cluster_sizes):.2f}")
    print(f"  Mediană: {sorted(cluster_sizes.values())[len(cluster_sizes)//2]}")

# Funcție pentru a afișa exemple din fiecare cluster
def display_cluster_examples(labels, n_examples=3):
    unique_labels = set(labels)
    for label in unique_labels:
        print(f"\nCluster {label}:")
        cluster_indices = np.where(labels == label)[0]
        sample_indices = np.random.choice(cluster_indices, size=min(n_examples, len(cluster_indices)), replace=False)
        for idx in sample_indices:
            article = df.iloc[idx]
            title = article['title'] if pd.notna(article['title']) and article['title'] != "No Title" else "Fără titlu"
            text = str(article['text']) if pd.notna(article['text']) else "Fără text"
            text_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"Index: {idx}, Titlu: {title}")
            print(f"Preview text: {text_preview}")
            print("---")

# Alegem numărul optim de clustere (de exemplu, 100)
chosen_n_clusters = 100
print(f"\nExemple de articole din clustere pentru {chosen_n_clusters} clustere:")
display_cluster_examples(results[chosen_n_clusters]['labels'])

# Creăm un dendrogram pentru a vizualiza structura ierarhică
plt.figure(figsize=(10, 7))
linkage_matrix = linkage(distance_matrix, method='average')
dendrogram(linkage_matrix)
plt.title('Dendrogram al Clusterizării Aglomerative')
plt.xlabel('Indice Articol')
plt.ylabel('Distanță')
plt.savefig('dendrogram.png')
plt.close()

print("\nDendrogramul a fost salvat ca 'dendrogram.png'")

# Salvăm rezultatele clusterizării
results_df = pd.DataFrame({
    n_clusters: results[n_clusters]['labels'] for n_clusters in n_clusters_list
})
results_df.to_csv('clustering_results.csv', index=False)
print("Rezultatele clusterizării au fost salvate în 'clustering_results.csv'")

# Salvăm informații detaliate despre clustere pentru numărul ales de clustere
chosen_labels = results[chosen_n_clusters]['labels']
cluster_info = []
for idx, label in enumerate(chosen_labels):
    article = df.iloc[idx]
    title = article['title'] if pd.notna(article['title']) and article['title'] != "No Title" else "Fără titlu"
    text = str(article['text']) if pd.notna(article['text']) else "Fără text"
    text_preview = text[:200] + "..." if len(text) > 200 else text
    cluster_info.append({
        'index': idx,
        'cluster': label,
        'title': title,
        'text_preview': text_preview,
        'author': str(article.get('author', 'Necunoscut')) if pd.notna(article.get('author')) else 'Necunoscut',
        'published_date': str(article.get('published', 'Necunoscut')) if pd.notna(article.get('published')) else 'Necunoscut'
    })

cluster_info_df = pd.DataFrame(cluster_info)
cluster_info_df.to_csv(f'cluster_info_{chosen_n_clusters}.csv', index=False)
print(f"Informații detaliate despre clustere au fost salvate în 'cluster_info_{chosen_n_clusters}.csv'")

# Afișăm distribuția mărimii clusterelor pentru numărul ales de clustere
chosen_cluster_sizes = results[chosen_n_clusters]['cluster_sizes']
plt.figure(figsize=(12, 6))
plt.bar(chosen_cluster_sizes.keys(), chosen_cluster_sizes.values())
plt.title(f'Distribuția mărimii clusterelor pentru {chosen_n_clusters} clustere')
plt.xlabel('Cluster')
plt.ylabel('Număr de articole')
plt.savefig(f'cluster_distribution_{chosen_n_clusters}.png')
plt.close()
print(f"Distribuția mărimii clusterelor a fost salvată ca 'cluster_distribution_{chosen_n_clusters}.png'")