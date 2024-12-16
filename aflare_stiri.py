import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from tqdm import tqdm

# Citim fișierul CSV
print("Citim fișierul CSV...")
df = pd.read_csv('output2.csv')

# Convertim coloana 'date' în format datetime
df['date'] = pd.to_datetime(df['date'])

# Funcție pentru a extrage cuvinte cheie din titlu
def extract_keywords(text):
    return set(word.lower() for word in text.split() if len(word) > 3)

# Funcție pentru a verifica dacă două știri au subiecte comune
def has_common_subject(headline1, headline2):
    keywords1 = extract_keywords(headline1)
    keywords2 = extract_keywords(headline2)
    return len(keywords1.intersection(keywords2)) > 0

# Funcție pentru clusterizare
def cluster_news(df, n_clusters=47):
    print("Clusterizăm știrile...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    return df, vectorizer

# Funcție pentru a găsi contradicții
def find_contradictions(df, tfidf, max_days_diff=7, similarity_threshold=0.3):
    contradictions = []
    
    print("Căutăm contradicții...")
    for i in tqdm(range(len(df))):
        for j in range(i+1, len(df)):
            date_diff = abs((df.iloc[i]['date'] - df.iloc[j]['date']).days)
            
            if date_diff <= max_days_diff and df.iloc[i]['cluster'] == df.iloc[j]['cluster']:
                if has_common_subject(df.iloc[i]['headlines'], df.iloc[j]['headlines']):
                    similarity = cosine_similarity(tfidf[i], tfidf[j])[0][0]
                    if similarity < similarity_threshold:
                        contradictions.append((df.iloc[i], df.iloc[j], similarity))
    
    return contradictions

# Clusterizăm știrile
df, vectorizer = cluster_news(df)

# Creăm matricea TF-IDF pentru toate știrile
tfidf_matrix = vectorizer.transform(df['text'])

# Găsim contradicții
contradictions = find_contradictions(df, tfidf_matrix)

# Afișăm rezultatele
print(f"\nS-au găsit {len(contradictions)} perechi de știri potențial contradictorii:")
for i, (news1, news2, similarity) in enumerate(contradictions, 1):
    print(f"\nContradicția {i}:")
    print(f"Știrea 1 ({news1['date'].date()}): {news1['headlines']}")
    print(f"Știrea 2 ({news2['date'].date()}): {news2['headlines']}")
    print(f"Cluster: {news1['cluster']}")
    print(f"Similaritate: {similarity:.4f}")
    print("-" * 50)

# Salvăm rezultatele într-un fișier
with open('contradictions_improved.txt', 'w', encoding='utf-8') as f:
    for i, (news1, news2, similarity) in enumerate(contradictions, 1):
        f.write(f"Contradicția {i}:\n")
        f.write(f"Știrea 1 ({news1['date'].date()}): {news1['headlines']}\n")
        f.write(f"Știrea 2 ({news2['date'].date()}): {news2['headlines']}\n")
        f.write(f"Cluster: {news1['cluster']}\n")
        f.write(f"Similaritate: {similarity:.4f}\n")
        f.write("-" * 50 + "\n")

print(f"Rezultatele au fost salvate în fișierul 'contradictions_improved.txt'")
