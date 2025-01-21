import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Încărcăm adnotările manuale și informațiile despre clustere
annotations_df = pd.read_csv('manual_annotations.csv')
cluster_info = pd.read_csv('cluster_info_100.csv')  # Presupunem că am folosit 100 de clustere

# Calculăm metricile pentru gruparea corectă (același eveniment)
same_event_true = annotations_df['same_event'].sum()
total_pairs = len(annotations_df)
correct_grouping_rate = same_event_true / total_pairs

# Calculăm metricile pentru detecția contradicțiilor
contradictions_true = annotations_df['contradictions'].sum()
contradiction_rate = contradictions_true / total_pairs

# Calculăm precizia clusterizării
def calculate_clustering_precision():
    correct_pairs = 0
    total_evaluated_pairs = 0
    for cluster in cluster_info['cluster'].unique():
        cluster_articles = cluster_info[cluster_info['cluster'] == cluster]
        cluster_indices = cluster_articles['index'].tolist()
        
        # Verificăm perechile adnotate din acest cluster
        cluster_annotations = annotations_df[
            (annotations_df['idx1'].isin(cluster_indices)) & 
            (annotations_df['idx2'].isin(cluster_indices))
        ]
        
        correct_pairs += cluster_annotations['same_event'].sum()
        total_evaluated_pairs += len(cluster_annotations)
    
    if total_evaluated_pairs > 0:
        return correct_pairs / total_evaluated_pairs
    else:
        return 0

clustering_precision = calculate_clustering_precision()

# Afișăm rezultatele
print("Analiza rezultatelor clusterizării:")
print(f"Total perechi adnotate: {total_pairs}")
print(f"Perechi despre același eveniment: {same_event_true} ({same_event_true/total_pairs:.2%})")
print(f"Perechi cu contradicții: {contradictions_true} ({contradictions_true/total_pairs:.2%})")
print(f"Rata de grupare corectă: {correct_grouping_rate:.2%}")
print(f"Precizia clusterizării: {clustering_precision:.2%}")
print(f"Rata de detecție a contradicțiilor: {contradiction_rate:.2%}")

# Calculăm metrici suplimentare pentru detecția contradicțiilor
# Presupunem că 'contradictions' este eticheta pozitivă
y_true = annotations_df['contradictions']
y_pred = annotations_df['same_event'] == False  # Presupunem că articolele care nu sunt despre același eveniment pot conține contradicții

conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nMetrici pentru detecția contradicțiilor:")
print(f"Matrice de confuzie:\n{conf_matrix}")
print(f"Precizie: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Analiza distribuției clusterelor
cluster_sizes = cluster_info['cluster'].value_counts()
print("\nAnaliza distribuției clusterelor:")
print(f"Număr total de clustere: {len(cluster_sizes)}")
print(f"Dimensiune medie cluster: {cluster_sizes.mean():.2f}")
print(f"Dimensiune mediană cluster: {cluster_sizes.median():.2f}")
print(f"Cel mai mare cluster: {cluster_sizes.max()} articole")
print(f"Cel mai mic cluster: {cluster_sizes.min()} articole")
