# Perceptron

This repository demonstrates the implementation and application of the **Perceptron Algorithm** for binary classification tasks. The Perceptron is a simple linear classifier used in machine learning, and this project explores its theoretical foundation and practical application.

## File Descriptions

- **Perceptron.ipynb**:  
  This Jupyter notebook contains a comprehensive walkthrough of the **Perceptron Algorithm**, including its theoretical basis, implementation, and application to real datasets.

### Outline:
- **Introduction**  
  Overview of the Perceptron Algorithm and its purpose in classification problems.
  
- **Algorithm**  
  Explanation of the mathematical model and learning rule behind the Perceptron.

- **Illustration**  
  Visualizations, including decision boundaries and error plots, to demonstrate the learning process.

- **Advantages and Disadvantages**  
  - **Advantages**:  
    Simplicity, efficiency for linearly separable data, and interpretability.  
  - **Disadvantages**:  
    Inability to solve non-linearly separable problems and sensitivity to feature scaling.

- **Code of Perceptron**  
  Detailed implementation of a **custom Perceptron class** with error tracking.

- **Applications on Data Sets**  
  Application of the Perceptron Algorithm on a student performance dataset.

---

## Dataset Used in Applications

### Student Performance Dataset
This dataset contains academic scores and demographic features of students. The goal is to classify students as **High Performance** or **Low Performance** based on their average test scores. 

#### Dataset Summary:
- **Source**: Collected for educational purposes.
- **Features**:
  - **Categorical**: Gender, race/ethnicity, parental level of education, lunch type, and test preparation course.
  - **Numerical**: Math score, reading score, writing score.
- **Target**:  
  A binary classification target created as:
  - `1` for High Performance (average score ≥ 70).
  - `0` for Low Performance (average score < 70).
  
The dataset is clean, containing no missing values, making it ideal for algorithm demonstration.

---

## Reproducibility

### **Dependencies**
The project requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `mlxtend`
- `scikit-learn`

### **Installation**
Install all required libraries using the following command:

```bash
pip install pandas numpy matplotlib mlxtend scikit-learn

