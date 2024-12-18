# Pokémon Type Classification

This repository demonstrates the implementation of a **neural network** using Stochastic Gradient Descent (SGD) to predict a Pokémon's primary type (**Type1**) based on its attributes. This project explores data preprocessing, model training, evaluation, and analysis of results, highlighting challenges such as class imbalance and feature limitations.

## File Descriptions

- **pokemon_classification.ipynb**:  
  This Jupyter notebook contains the full pipeline for predicting Pokémon `Type1`. It includes data preprocessing, neural network implementation, training, evaluation, and visualization of results.

### Outline:
- **Introduction**  
  Overview of the project goal: Predicting a Pokémon's primary type using machine learning techniques.

- **Data Cleaning and Preprocessing**  
  - Handling missing values.  
  - One-hot encoding of categorical features like `Type2`.  
  - Preparation of input (features) and output (labels).

- **Model Training**  
  Implementation of a simple feedforward neural network trained with **Stochastic Gradient Descent (SGD)**.

- **Evaluation and Results**  
  - Performance analysis using accuracy and loss metrics.  
  - Visualizations of training and validation accuracy/loss curves.  
  - Discussion of model limitations and potential improvements.

- **Challenges**  
  - Insufficient features leading to low performance.  
  - Overfitting of training data.  
  - Class imbalance in the target variable (`Type1`).

---

## Dataset Used

### Pokémon Dataset
The dataset includes Pokémon attributes such as name, type classifications, and evolutionary stages.

#### Dataset Summary:
- **Source**: Pokémon data collected for educational purposes.  
- **Features**:
  - `Name`: Name of the Pokémon.
  - `Type1`: Primary type of the Pokémon (target variable).
  - `Type2`: Secondary type of the Pokémon (optional feature).  
  - `Evolution`: Next evolutionary stage (used as an additional feature).  

- **Target**:  
  The primary type of the Pokémon, `Type1`, consists of 18 possible classes (e.g., Grass, Fire, Water).

---

## Reproducibility

### **Dependencies**
The project requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

### **Installation**
Install all required libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
