import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm


"""
cercare modello giusto, cambiare randomForestClassifier con quello che funzioni
sklearn.feature_selection?
linear SVC?

X = riga fatta da (tutto tranne ID e Menthal_Health_Condition)
Age,Gender,Sleep_Hours_Per_Day,Work_Hours_Per_Week,Exercise_Frequency_Per_Week,
Mood_Score (0-100),Anxiety_Score (0-100),Depression_Score (0-100),Stress_Score (0-100),
Energy_Level (0-100),Social_Interaction (1-10),Concentration (1-10),Appetite (1-10),
Life_Satisfaction (0-100),Therapy_Sessions_Per_Month,Medication,Family_History,
Hallucinations,Manic_Episodes_Per_Year
y = Menthal_Health_Condition

Seguendo il grafico su https://scikit-learn.org/stable/machine_learning_map.html
ho scelto LinearSVC

"""

df = pd.read_csv('mental_health_dataset.csv')

#df.drop("Patient_ID", axis=1)

# Encode colonne categoriche

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Target encode
le_target = LabelEncoder()
df['Mental_Health_Condition'] = le_target.fit_transform(df['Mental_Health_Condition'])

# Features e target
FEATURES = [
    'Age', 'Gender', 'Sleep_Hours_Per_Day', 'Work_Hours_Per_Week',
    'Exercise_Frequency_Per_Week', 'Mood_Score (0-100)', 'Anxiety_Score (0-100)',
    'Depression_Score (0-100)', 'Stress_Score (0-100)', 'Energy_Level (0-100)',
    'Social_Interaction (1-10)', 'Concentration (1-10)', 'Appetite (1-10)',
    'Life_Satisfaction (0-100)', 'Therapy_Sessions_Per_Month', 'Medication',
    'Family_History', 'Hallucinations', 'Manic_Episodes_Per_Year'
]

X = df[FEATURES]
y = df['Mental_Health_Condition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creiamo i train e test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # mantiene proporzione classi
)

# Creare l'oggetto modello e dargli i set di train
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
y_pred = lin_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2f}")

report = classification_report(y_test, y_pred, target_names=le_target.classes_)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)
print("matrice di confusioneo matrice di errore\n",cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
 xticklabels=le_target.classes_,
 yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#gridsearch
"""
veloce    
    
param_grid_semplice = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
}
# Solo 12 combinazioni, usa i default per il resto

grid = GridSearchCV(
    svm.LinearSVC(max_iter=10000, random_state=42),
    param_grid_semplice,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
print("Best:", grid.best_params_, "→", grid.best_score_)


"""
# lento
param_grid = [
    # ── Combinazione 1: L2 + squared_hinge (la standard) ──
    {
        'C':            [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty':      ['l2'],
        'loss':         ['squared_hinge'],
        'dual':         [True, False],       # entrambi validi qui
        'class_weight': [None, 'balanced'],
    },
    # ── Combinazione 2: L2 + hinge (richiede dual=True) ──
    {
        'C':            [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty':      ['l2'],
        'loss':         ['hinge'],
        'dual':         [True],              # obbligatorio con hinge
        'class_weight': [None, 'balanced'],
    },
    # ── Combinazione 3: L1 + squared_hinge (richiede dual=False) ──
    # L1 fa anche feature selection: azzera i coefficienti inutili
    {
        'C':            [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty':      ['l1'],
        'loss':         ['squared_hinge'],
        'dual':         [False],             # obbligatorio con l1
        'class_weight': [None, 'balanced'],
    },
]
# Totale: (6×2×2) + (6×1×2) + (6×1×2) = 24+12+12 = 48 combinazioni
# Con cv=5 → 48 × 5 = 240 addestramenti (LinearSVC è veloce, ok!)

# ─────────────────────────────────────────────────────────────
# GRIDSEARCH
# ─────────────────────────────────────────────────────────────
grid = GridSearchCV(
    estimator=svm.LinearSVC(max_iter=10000, random_state=42),
    # max_iter alto → evita warning di non-convergenza
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation per ogni combinazione
    scoring='accuracy',      # metrica da massimizzare
    n_jobs=-1,               # usa tutti i core della CPU (parallelo)
    verbose=1                # stampa il progresso
)

# ATTENZIONE: X_train deve essere GIÀ SCALATO (StandardScaler)!
# LinearSVC è molto sensibile alla scala delle feature.
grid.fit(X_train, y_train)


# Risultati

print("\n" + "="*60)
print("RISULTATI GRIDSEARCH")
print("="*60)
print(f"Migliori parametri:  {grid.best_params_}")
print(f"Best CV accuracy:    {grid.best_score_:.4f}")

# Valutazione finale sul test set (dati mai visti)
best_model = grid.best_estimator_       # già riaddestrato sui parametri ottimi
y_pred = best_model.predict(X_test)

print(f"Test accuracy:       {accuracy_score(y_test, y_pred):.4f}")
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))