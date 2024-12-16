import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Citim fișierul CSV
print("Citim fișierul CSV...")
df = pd.read_csv('output.csv')

# Inițializăm TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=100)  # Limitam la 100 de caracteristici pentru a reduce dimensiunea

# Funcție pentru a crea embedding-uri pentru un text
def create_embedding(text):
    if isinstance(text, str):
        return vectorizer.transform([text]).toarray()[0]
    else:
        return np.zeros(100)  # Returnăm un vector de zerouri pentru valori non-string

# Aplicăm vectorizarea pe coloanele 'text' și 'ctext'
print("Antrenăm vectorizatorul pe coloana 'text'...")
vectorizer.fit(df['text'].fillna(''))

print("Generăm embedding-uri pentru coloana 'text'...")
df['text_embedding'] = df['text'].apply(create_embedding)

print("Generăm embedding-uri pentru coloana 'ctext'...")
df['ctext_embedding'] = df['ctext'].apply(create_embedding)

# Convertim embedding-urile în string pentru a le putea salva în CSV
df['text_embedding'] = df['text_embedding'].apply(lambda x: ' '.join(map(str, x)))
df['ctext_embedding'] = df['ctext_embedding'].apply(lambda x: ' '.join(map(str, x)))

# Salvăm rezultatul într-un nou fișier CSV
print("Salvăm rezultatele în output2.csv...")
df.to_csv('output2.csv', index=False)

print("Procesul a fost finalizat cu succes!")
