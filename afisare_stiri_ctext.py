import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from tqdm import tqdm

# Citim fișierul CSV
print("Citim fișierul CSV...")
df = pd.read_csv('output2.csv')

# Convertim coloana 'date' în format datetime
df['date'] = pd.to_datetime(df['date'])

# Funcție pentru a converti string-ul embedding în array numpy
def string_to_array(s):
    return np.fromstring(s.strip('[]'), sep=' ')

# Convertim embedding-urile din string în array numpy
print("Procesăm embedding-urile...")
df['text_embedding'] = df['text_embedding'].apply(string_to_array)
df['ctext_embedding'] = df['ctext_embedding'].apply(string_to_array)

# Funcție îmbunătățită pentru a verifica dacă două știri au subiecte comune
def has_common_subject(headline1, headline2):
    words1 = set(word.lower() for word in headline1.split() if len(word) > 3)
    words2 = set(word.lower() for word in headline2.split() if len(word) > 3)
    return len(words1.intersection(words2)) > 1  # Cerință mai puțin strictă

# Funcție pentru a detecta contradicții în conținut
def detect_contradiction(text1, text2):
    contradictory_pairs = [
        ('launches', 'halts'), ('starts', 'stops'), ('begins', 'ends'),
        ('approves', 'rejects'), ('increases', 'decreases'), ('supports', 'opposes')
    ]
    words1 = set(word.lower() for word in text1.split())
    words2 = set(word.lower() for word in text2.split())
    
    for pair in contradictory_pairs:
        if (pair[0] in words1 and pair[1] in words2) or (pair[1] in words1 and pair[0] in words2):
            return True
    return False

# Funcție pentru a găsi contradicții
def find_contradictions(df, max_days_diff=30, similarity_threshold=0.5):
    contradictions = set()
    
    print("Căutăm contradicții...")
    for i in tqdm(range(len(df))):
        for j in range(i+1, len(df)):
            date_diff = abs((df.iloc[i]['date'] - df.iloc[j]['date']).days)
            
            if date_diff <= max_days_diff:
                if has_common_subject(df.iloc[i]['headlines'], df.iloc[j]['headlines']):
                    similarity_text = cosine_similarity([df.iloc[i]['text_embedding']], [df.iloc[j]['text_embedding']])[0][0]
                    similarity_ctext = cosine_similarity([df.iloc[i]['ctext_embedding']], [df.iloc[j]['ctext_embedding']])[0][0]
                    
                    avg_similarity = (similarity_text + similarity_ctext) / 2
                    
                    if avg_similarity > similarity_threshold or detect_contradiction(df.iloc[i]['text'], df.iloc[j]['text']):
                        contradictions.add((min(i, j), max(i, j), avg_similarity))
    
    return [(df.iloc[i], df.iloc[j], sim) for i, j, sim in contradictions]

# Găsim contradicții
contradictions = find_contradictions(df)

# Funcție pentru a formata o știre pentru afișare
def format_news(news):
    return f"""
Autor: {news['author']}
Data: {news['date'].date()}
Titlu: {news['headlines']}
Link: {news['read_more']}
Text: {news['text']}
Context: {news['ctext']}
Cluster: {news['cluster']}
"""

# Afișăm și salvăm rezultatele
print(f"\nS-au găsit {len(contradictions)} perechi de știri potențial contradictorii:")

with open('contradictions_improved.txt', 'w', encoding='utf-8') as f:
    for i, (news1, news2, similarity) in enumerate(contradictions, 1):
        output = f"\nContradicția {i}:\n"
        output += "Știrea 1:" + format_news(news1)
        output += "\nȘtirea 2:" + format_news(news2)
        output += f"Similaritate medie: {similarity:.4f}\n"
        output += "-" * 50 + "\n"
        
        print(output)
        f.write(output)

print(f"Rezultatele au fost salvate în fișierul 'contradictions_improved.txt'")
