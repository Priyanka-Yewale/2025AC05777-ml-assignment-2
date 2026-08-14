import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

st.title("🍷 Wine Quality Classification Web App")
st.markdown("Interactive demonstration of Machine Learning classification models for predicting wine quality.")

# ---------------------------------------------------------
# 1. Train and Cache Models Live to Avoid Version Mismatches
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_models():
    # Load raw dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    
    # Binary target (>= 6 is good quality)
    df['quality_label'] = (df['quality'] >= 6).astype(int)
    df = df.drop(columns=['quality'])
    
    X = df.drop(columns=['quality_label'])
    y = df['quality_label']
    
    # Split 80/20 with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Fit all models on scaled training data
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
    return trained_models

# Train models on app startup
trained_models = load_and_train_models()

# ---------------------------------------------------------
# 2. Sidebar & Interface Logic
# ---------------------------------------------------------
st.sidebar.header("1. Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
)

if uploaded_file is not None:
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Uploaded Test Data Preview")
    st.dataframe(data.head())

    if 'target' not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column!")
    else:
        X_test = data.drop(columns=['target'])
        y_test = data['target']

        # Get the selected trained model
        model = trained_models[model_choice]

        # Predict
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred

        # Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"📈 Evaluation Metrics for {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("AUC", f"{auc:.3f}")
        col3.metric("Precision", f"{prec:.3f}")
        col4.metric("Recall", f"{rec:.3f}")
        col5.metric("F1 Score", f"{f1:.3f}")
        col6.metric("MCC", f"{mcc:.3f}")

        # Visualizations
        st.write("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with col_right:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose())
else:
    st.info("👈 Please upload `test_data.csv` using the sidebar to view evaluations.")
