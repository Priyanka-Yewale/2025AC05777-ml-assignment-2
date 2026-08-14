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
| **Logistic Regression** | 0.7406 | 0.8242 | 0.7683 | 0.7368 | 0.7522 | 0.4808 |
| **Decision Tree** | 0.7531 | 0.7513 | 0.7644 | 0.7778 | 0.7710 | 0.5034 |
| **KNN** | 0.7406 | 0.8117 | 0.7588 | 0.7544 | 0.7566 | 0.4790 |
| **Naive Bayes** | 0.7219 | 0.7884 | 0.7733 | 0.6784 | 0.7227 | 0.4500 |
| **Random Forest (Ensemble)** | **0.8063** | **0.9018** | **0.8344** | **0.7953** | **0.8144** | **0.6128** |

### Performance Observations

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundaries handle normalized continuous chemical features well, providing a reliable linear baseline. |
| **Decision Tree** | Captures non-linear thresholds easily, achieving strong balanced accuracy and positive correlation metrics on scaled features. |
| **KNN** | Sensitive to distance scale; proper feature standardization ensures balanced Euclidean distance calculations and reliable classification. |
| **Naive Bayes** | Assumes feature independence which slightly lowers precision and recall on correlated chemical features. |
| **Random Forest (Ensemble)** | Top performer across all metrics; decision tree ensemble effectively minimizes variance and maximizes AUC (0.9018) and MCC (0.6128). |
| **Overall Winner** | **Random Forest (Ensemble)** |
