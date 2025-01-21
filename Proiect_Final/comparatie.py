import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Încărcăm modelul salvat
print("Încărcăm modelul...")
model = load('contradiction_detection_model.joblib')

# Încărcăm setul de date independent
print("Încărcăm setul de date independent...")
independent_data = pd.read_csv('independent_test_data.csv')  # Asigurați-vă că aveți acest fișier

# Funcție pentru a combina textele a două articole
def combine_texts(row):
    return f"{row['text1']} [SEP] {row['text2']}"

# Pregătim datele pentru predicție
print("Pregătim datele pentru predicție...")
X_independent = independent_data.apply(combine_texts, axis=1)
y_independent = independent_data['contradictions']

# Facem predicții
print("Facem predicții pe setul de date independent...")
y_pred = model.predict(X_independent)
y_prob = model.predict_proba(X_independent)[:, 1]  # Probabilitatea pentru clasa pozitivă (contradicție)

# Evaluăm performanța
print("\nPerformanța modelului pe setul de date independent:")
print(classification_report(y_independent, y_pred))

print("\nMatricea de confuzie:")
print(confusion_matrix(y_independent, y_pred))

# Analiza erorilor
errors = independent_data[y_independent != y_pred].copy()
errors['predicted_prob'] = y_prob[y_independent != y_pred]
print(f"\nNumăr total de erori: {len(errors)}")

# Afișăm câteva exemple de erori
print("\nExemple de predicții incorecte:")
for idx, row in errors.sort_values('predicted_prob', ascending=False).head().iterrows():
    print(f"\nText 1: {row['text1'][:100]}...")
    print(f"Text 2: {row['text2'][:100]}...")
    print(f"Etichetă reală: {'Contradicție' if row['contradictions'] else 'Nu este contradicție'}")
    print(f"Predicție: {'Contradicție' if y_pred[idx] else 'Nu este contradicție'}")
    print(f"Probabilitate de contradicție: {row['predicted_prob']:.2f}")

# Analiză detaliată a performanței
thresholds = np.arange(0, 1.1, 0.1)
for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    accuracy = (y_pred_threshold == y_independent).mean()
    print(f"\nPrag de decizie: {threshold:.1f}")
    print(f"Acuratețe: {accuracy:.2f}")
    print(classification_report(y_independent, y_pred_threshold))

# Salvăm rezultatele predicțiilor
independent_data['predicted_contradiction'] = y_pred
independent_data['contradiction_probability'] = y_prob
independent_data.to_csv('independent_data_with_predictions.csv', index=False)
print("\nRezultatele predicțiilor au fost salvate în 'independent_data_with_predictions.csv'")

# Analiza cazurilor cu probabilitate ridicată, dar etichetă diferită
high_prob_threshold = 0.8
low_prob_threshold = 0.2
high_prob_errors = errors[errors['predicted_prob'] >= high_prob_threshold]
low_prob_errors = errors[errors['predicted_prob'] <= low_prob_threshold]

print(f"\nCazuri cu probabilitate ridicată (>={high_prob_threshold}) dar etichetă incorectă:")
print(high_prob_errors[['text1', 'text2', 'contradictions', 'predicted_prob']].head())

print(f"\nCazuri cu probabilitate scăzută (<={low_prob_threshold}) dar etichetă incorectă:")
print(low_prob_errors[['text1', 'text2', 'contradictions', 'predicted_prob']].head())

# Distribuția lungimii textelor pentru predicții corecte vs. incorecte
independent_data['text_length'] = independent_data.apply(lambda row: len(row['text1']) + len(row['text2']), axis=1)
correct_predictions = independent_data[y_independent == y_pred]
incorrect_predictions = independent_data[y_independent != y_pred]

print("\nDistribuția lungimii textelor:")
print("Pentru predicții corecte:")
print(correct_predictions['text_length'].describe())
print("\nPentru predicții incorecte:")
print(incorrect_predictions['text_length'].describe())
