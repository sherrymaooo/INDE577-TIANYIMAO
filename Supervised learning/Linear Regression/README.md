# Student Math Score Prediction Using Regression Models  

This project implements multiple **linear regression models** to predict students' math scores based on various demographic, parental, and academic factors. The models include **Normal Equation**, **Ridge Regression**, **Lasso Regression**, and **ElasticNet Regression**. These techniques are applied to evaluate and compare their effectiveness in predicting student performance.  

---

## **Introduction**  

Linear regression models are supervised machine learning techniques used to predict a continuous outcome variable based on input features.  
- **Normal Equation** provides an exact solution without regularization.  
- **Ridge Regression** applies **L2 regularization** to shrink large coefficients and reduce overfitting.  
- **Lasso Regression** uses **L1 regularization** for automatic feature selection by shrinking some coefficients to zero.  
- **ElasticNet Regression** combines both L1 and L2 regularization for a balanced approach.  

In this project, we predict **math scores** using features such as gender, parental education level, test preparation courses, and other factors.  

---

## **Dataset**  

The dataset used is the **Students Performance Dataset**, which includes various academic and demographic attributes.  

- **Target Variable**: `math score` (continuous numerical value)  
- **Features**:  
   - Reading score  
   - Writing score  
   - Gender  
   - Race/Ethnicity  
   - Parental level of education  
   - Lunch type  
   - Test preparation course  
- **Size**: 1,000 rows after preprocessing  
- **Data Source**: This dataset is commonly used for student performance prediction tasks.  

---

## **Workflow Outline**  

The project is structured into four main steps for clarity and reproducibility:  

### **Step 1: Data Cleaning and Analysis**  
- Inspect the dataset and handle missing values.  
- Encode categorical variables into numerical format using **One-Hot Encoding**.  
- Standardize numerical features using **Z-score normalization** to ensure consistent feature scaling.  
- Generate visualizations, including:  
   - Target variable distribution.  
   - Correlation heatmaps to identify relationships between features.  

### **Step 2: Train the Models**  
- Split the data into training and testing sets (70/30 split).  
- Train the following regression models:  
  1. **Normal Equation** (no regularization)  
  2. **Ridge Regression** (L2 regularization)  
  3. **Lasso Regression** (L1 regularization)  
  4. **ElasticNet Regression** (L1 + L2 regularization).  

### **Step 3: Evaluate the Models**  
- Use the test set to make predictions.  
- Evaluate model performance using the following metrics:  
   - **Mean Squared Error (MSE)**: Average squared difference between actual and predicted values.  
   - **R-squared (\( R^2 \))**: Proportion of variance explained by the model.  
- Visualize results:  
   - **Prediction vs Actual Scatter Plots**: Shows alignment between actual and predicted scores.  
   - **Residual Analysis**: Evaluate error distributions for each model.  
   - **MSE Comparison**: Compare error magnitudes across models.  

### **Step 4: Visualize 3D Predictions**  
- Generate **3D scatter plots** for all models (Normal Equation, Ridge, Lasso, ElasticNet) to compare predicted and actual math scores using two standardized features.  
- Analyze the alignment of predictions with actual scores to evaluate model accuracy visually.  

## **Reproducibility**  

To reproduce the results, follow these steps:  

1. **Dependencies**: Ensure the following libraries are installed:  
   ```bash  
   pip install pandas numpy scikit-learn matplotlib seaborn  
