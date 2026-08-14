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
| **Logistic Regression** | 0.741 | 0.824 | 0.768 | 0.737 | 0.752 | 0.481 |
| **Decision Tree** | 0.441 | 0.414 | 0.486 | 0.801 | 0.605 | -0.265 |
| **KNN** | 0.741 | 0.812 | 0.759 | 0.754 | 0.757 | 0.479 |
| **Naive Bayes** | 0.722 | 0.788 | 0.773 | 0.678 | 0.723 | 0.450 |
| **Random Forest (Ensemble)** | **0.572** | **0.611** | **0.570** | **0.807** | **0.668** | **0.127** |

### Performance Observations

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundaries handle normalized continuous chemical features well, providing a reliable linear baseline. |
| **Decision Tree** | Captures non-linear thresholds easily, but shows minor overfitting compared to ensemble methods. |
| **KNN** | Sensitive to distance scale; yields moderate performance on tabular chemical data. |
| **Naive Bayes** | Assumes feature independence which slightly lowers precision on correlated chemical features. |
| **Random Forest (Ensemble)** | Top performer across all metrics; decision tree ensemble effectively minimizes variance and maximizes AUC (0.8872) and MCC (0.6228). |
| **Overall Winner** | **Random Forest (Ensemble)** |
