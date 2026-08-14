import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 1. Load Dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Convert quality rating into binary target (1: High Quality, 0: Low Quality)
df['quality_label'] = (df['quality'] >= 6).astype(int)
df = df.drop(columns=['quality'])

# Features & Target
X = df.drop(columns=['quality_label'])
y = df['quality_label']

# 2. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test dataset for Streamlit app
test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['target'] = y_test.values
test_df.to_csv("test_data.csv", index=False)

# Make directory for models
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# 3. Models Definition
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 4. Train ALL 5 models on scaled training data
for name, model in models.items():
    # Fit EVERY model on X_train_scaled
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Save model
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.pkl")

    # Metrics
    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }
    results.append(metrics)

# Print comparison table
results_df = pd.DataFrame(results)
print("\n--- MODEL PERFORMANCE METRICS ---")
print(results_df.to_string(index=False))
