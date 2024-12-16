import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Citirea fișierului CSV
input_file = "news_summary_augmented.csv"
data = pd.read_csv(input_file)

# Verificarea coloanelor
required_columns = ["author", "date", "headlines", "read_more", "text", "ctext"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Fișierul CSV trebuie să conțină coloanele: {', '.join(required_columns)}")

# Selectarea coloanei pentru procesare (text sau ctext)
selected_column = "text"  # Poți schimba în "ctext" dacă e necesar

# Transformarea textului în reprezentare numerică folosind TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data[selected_column].fillna(""))

# Aplicarea algoritmului de clusterizare K-Means
num_clusters = 500  # Specifică numărul de clustere dorit, radical din 4500
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data["cluster"] = kmeans.fit_predict(tfidf_matrix)

# Scrierea rezultatelor într-un fișier CSV de ieșire
output_file = "output.csv"
data.to_csv(output_file, index=False)

print(f"Clusterizarea a fost finalizată. Rezultatele au fost salvate în {output_file}.")