# Wine Quality Classification & Streamlit App

## a. Problem Statement
The objective of this project is to construct, evaluate, and compare multiple supervised Machine Learning classification models to predict whether red wine is of high quality based on continuous chemical attributes.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository (Wine Quality Dataset)
- **Instances**: 1,599 samples
- **Features**: 12 - 11 continuous chemical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) + 1 engineered binary target (`1` for quality ≥ 6, `0` otherwise).

## c. GitHub Repository Link
https://github.com/Priyanka-Yewale/2025AC05777-ml-assignment-2/

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7438 | 0.8112 | 0.7612 | 0.7515 | 0.7563 | 0.4851 |
| **Decision Tree** | 0.7469 | 0.7454 | 0.7515 | 0.7697 | 0.7605 | 0.4908 |
| **KNN** | 0.7281 | 0.7854 | 0.7383 | 0.7515 | 0.7448 | 0.4533 |
| **Naive Bayes** | 0.7188 | 0.7892 | 0.7423 | 0.7212 | 0.7316 | 0.4361 |
| **Random Forest (Ensemble)** | **0.8125** | **0.8872** | **0.8253** | **0.8242** | **0.8247** | **0.6228** |

### Performance Observations

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundaries handle normalized continuous chemical features well, providing a reliable linear baseline. |
| **Decision Tree** | Captures non-linear thresholds easily, but shows minor overfitting compared to ensemble methods. |
| **KNN** | Sensitive to distance scale; yields moderate performance on tabular chemical data. |
| **Naive Bayes** | Assumes feature independence which slightly lowers precision on correlated chemical features. |
| **Random Forest (Ensemble)** | Top performer across all metrics; decision tree ensemble effectively minimizes variance and maximizes AUC (0.8872) and MCC (0.6228). |
| **Overall Winner** | **Random Forest (Ensemble)** |
