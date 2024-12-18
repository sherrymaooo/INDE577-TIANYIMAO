# Logistic Regression

This repository demonstrates the implementation of Logistic Regression to predict students' test preparation course completion status. Logistic Regression, a powerful yet interpretable machine learning model, serves as the foundation for this analysis.

---

## File Descriptions
**"Logistic_Regression.ipynb"** contains the description and implementation of Logistic Regression, including all preprocessing steps to prepare the data for modeling.

---

## Knowledge About Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification tasks. It models the relationship between the binary dependent variable and one or more independent variables using the logistic (sigmoid) function. It outputs probabilities to predict the likelihood of class membership.
---

## Dataset Used
The dataset includes information about students' academic performance and other demographics. The target variable, `test preparation course`, indicates whether students completed the course.

### *Students Performance Dataset*
**Source**: Custom dataset with the following features:

### Key Features:
1. **Math Score**: Numerical.
2. **Reading Score**: Numerical.
3. **Writing Score**: Numerical.
4. **Gender**: Categorical (encoded).
5. **Lunch Type**: Categorical (encoded).
6. **Parental Education Level**: Categorical (encoded).
7. **Test Preparation Course**: Binary target variable (0 = Incomplete, 1 = Completed).

### Preprocessing Steps:
- Categorical features were encoded using **One-Hot Encoding**.
- Numerical features were scaled using **StandardScaler**.

---

## Goals and Workflow

The primary goal of this project is to predict whether a student completed the test preparation course using Logistic Regression. The workflow includes:

1. **Data Cleaning and Preprocessing**
2. **Feature Engineering**
3. **Training the Logistic Regression Model**
4. **Evaluating Model Performance**

---

### Summary:
The Logistic Regression model demonstrated strong performance as a baseline classifier for predicting the target variable.

---

## Reproducibility

To reproduce the results of this project, follow these steps:

### Prerequisites
Ensure you have Python and the required libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the Code
Execute the notebook "Logistic_Regression.ipynb" step by step to reproduce the results.

