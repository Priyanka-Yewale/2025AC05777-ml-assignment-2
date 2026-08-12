import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

st.title("🍷 Wine Quality Classification Web App")
st.markdown("Interactive demonstration of Machine Learning classification models for predicting wine quality.")

# Sidebar Configuration
st.sidebar.header("1. Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
)

model_mapping = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

if uploaded_file is not None:
    # Explicitly reset file pointer and read CSV safely
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Uploaded Test Data Preview")
    st.dataframe(data.head())
    if 'target' not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column!")
    else:
        X_test = data.drop(columns=['target'])
        y_test = data['target']

        # Load Model
        model_path = model_mapping[model_choice]
        model = joblib.load(model_path)

        # Predict
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred

        # Metrics
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
