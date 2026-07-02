import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



"""
AdaBoostClassifier Accuratezza del modello: 0.62
con grid serarch Best CV accuracy:    0.9625
                 Test accuracy:       0.9500
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


# Creiamo i train e test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # mantiene proporzione classi
)

# Creare l'oggetto modello e dargli i set di train
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2f}")

report = classification_report(y_test, y_pred, target_names=le_target.classes_)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)
print("matrice di confusioneo matrice di errore\n",cm)

#gridsearch
#da fare  i parametri
param_grid = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],

    # Modello base con diverse profondità:
    # depth=1 → stump classico (default)
    # depth=2,3 → alberi leggermente più espressivi
    #             (spesso migliora MOLTO su problemi multiclasse!)
    'estimator': [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=3),
    ]
}
# ─────────────────────────────────────────────────────────────
# GRIDSEARCH
# ─────────────────────────────────────────────────────────────
grid = GridSearchCV(
    estimator=AdaBoostClassifier(n_estimators=100, random_state=42),
    # max_iter alto → evita warning di non-convergenza
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation per ogni combinazione
    scoring='accuracy',      # metrica da massimizzare
    n_jobs=-1,               # usa tutti i core della CPU (parallelo)
    verbose=1                # stampa il progresso
)


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

plt.figure(figsize=(6,4))  #crea un errore quando chiudi la finestra del grafico se non la fai non va avanti
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
 xticklabels=le_target.classes_,
 yticklabels=le_target.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()