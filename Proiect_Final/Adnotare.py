import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Încărcăm rezultatele clusterizării și informațiile despre articole
cluster_info = pd.read_csv('cluster_info_100.csv')  # Presupunem că am ales 100 de clustere
df = pd.read_pickle('news_articles_with_embeddings.pkl')

# Funcție pentru a selecta perechi de articole din același cluster
def select_pairs_from_cluster(cluster_label, n_pairs=3):
    cluster_articles = cluster_info[cluster_info['cluster'] == cluster_label]
    if len(cluster_articles) < 2:
        return []
    pairs = []
    for _ in range(min(n_pairs, len(cluster_articles) // 2)):
        pair = cluster_articles.sample(n=2)['index'].tolist()
        pairs.append(pair)
    return pairs

# Selectăm perechi de articole din fiecare cluster
all_pairs = []
for cluster in tqdm(cluster_info['cluster'].unique(), desc="Selectare perechi"):
    pairs = select_pairs_from_cluster(cluster)
    all_pairs.extend(pairs)

random.shuffle(all_pairs)  # Amestecăm perechile pentru a evita bias-ul

# Funcție pentru a gestiona text-ul articolului
def get_article_text(article, max_length=200):
    if isinstance(article['text'], float):
        return "Text indisponibil"
    text = str(article['text'])
    return text[:max_length] + "..." if len(text) > max_length else text

# Funcție pentru adnotare manuală
def annotate_pair(pair):
    idx1, idx2 = pair
    article1 = df.iloc[idx1]
    article2 = df.iloc[idx2]
    
    print("\n--- Pereche de articole ---")
    print(f"Articol 1:")
    print(f"Titlu: {article1['title']}")
    print(f"Text: {get_article_text(article1)}")
    print("\nArticol 2:")
    print(f"Titlu: {article2['title']}")
    print(f"Text: {get_article_text(article2)}")
    
    same_event = input("Sunt despre același eveniment? (da/nu): ").lower() == 'da'
    contradictions = input("Conțin contradicții? (da/nu): ").lower() == 'da'
    
    return {
        'idx1': idx1,
        'idx2': idx2,
        'same_event': same_event,
        'contradictions': contradictions
    }

# Procesul de adnotare
print("Începe procesul de adnotare...")
annotations = []
for pair in tqdm(all_pairs, desc="Adnotare"):
    try:
        annotation = annotate_pair(pair)
        annotations.append(annotation)
    except Exception as e:
        print(f"Eroare la adnotarea perechii {pair}: {e}")
        continue
    
    # Întrebăm utilizatorul dacă dorește să continue
    if input("Continuați adnotarea? (da/nu): ").lower() != 'da':
        break

# Salvăm adnotările
annotations_df = pd.DataFrame(annotations)
annotations_df.to_csv('manual_annotations.csv', index=False)
print("Adnotările au fost salvate în 'manual_annotations.csv'")

# Afișăm statistici despre adnotări
total_annotations = len(annotations_df)
same_event_count = annotations_df['same_event'].sum()
contradictions_count = annotations_df['contradictions'].sum()

print(f"\nStatistici adnotări:")
print(f"Total perechi adnotate: {total_annotations}")
print(f"Perechi despre același eveniment: {same_event_count} ({same_event_count/total_annotations:.2%})")
print(f"Perechi cu contradicții: {contradictions_count} ({contradictions_count/total_annotations:.2%})")