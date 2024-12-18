# Student Depression Prediction Using K-Nearest Neighbors (KNN)

This project implements the **K-Nearest Neighbors (KNN)** algorithm to predict depression status among students based on mental health and lifestyle data. KNN is a simple, non-parametric, supervised machine learning model that predicts outcomes by identifying the "k" closest neighbors to a query point.

---

## **Introduction**  
KNN works by calculating the distances between a query point and all other points in the dataset.  
- For **classification** tasks, the label of a new data point is determined by a majority vote among its k-nearest neighbors.  
- For **regression** tasks, the outcome is the average value of its k-nearest neighbors.

In this project, KNN is applied to predict **depression status** (0 = Not Depressed, 1 = Depressed). The algorithm uses **Euclidean distance** as the metric for measuring closeness between data points:

\[
d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
\]

---

## **Dataset**  

The dataset used in this project is the **Student Depression Dataset**, which contains information about students' mental health and related factors. It includes a mix of numerical and categorical features relevant to depression prediction.

- **Target Variable**: `Depression` (binary: 0 = Not Depressed, 1 = Depressed)  
- **Features**:  
   - Age  
   - Gender  
   - CGPA (Cumulative Grade Point Average)  
   - Work Pressure  
   - Study Satisfaction  
   - Sleep Duration  
   - Dietary Habits  
   - Degree  
   - Suicidal Thoughts History  
   - Family History of Mental Illness  
- **Size**: Approximately 11,000 rows after preprocessing  
- **Data Source**: The dataset represents mental health patterns in students and is designed for classification tasks.

---

## **Workflow Outline**  

The project is divided into the following steps to ensure clear and reproducible results:

### **Step 1: Data Cleaning and Analysis**  
- Remove irrelevant features (e.g., IDs, Cities).  
- Encode categorical variables into numerical values using **Label Encoding**.  
- Handle missing values using the median of each feature.  
- Standardize the feature values using **StandardScaler** to ensure consistent distance measurements.  

### **Step 2: Train the KNN Model**  
- Split the data into training and testing sets using a 60/40 split.  
- Train the KNN model using an initial value of \( k = 5 \).

### **Step 3: Evaluate the Model**  
- Predict depression status on the test data.  
- Evaluate the model using:  
  - **Accuracy**: Proportion of correct predictions.  
  - **Confusion Matrix**: Visual representation of correct and incorrect classifications.  
  - **Classification Report**: Provides precision, recall, and F1-score for both classes.  
- Plot the confusion matrix for intuitive interpretation.

### **Step 4: Optimize k**  
- Evaluate the model across a range of \( k \) values (e.g., 1 to 20).  
- Plot the **Error Rate vs. k** graph to identify the value of \( k \) that minimizes error and ensures optimal performance.

---

## **Advantages and Disadvantages of KNN**

**Advantages:**  
- KNN does not require a training phase, making it a **lazy learning algorithm**.  
- Simple to implement and requires minimal computation on smaller datasets.  
- No assumptions about the data distribution are needed.

**Disadvantages:**  
- KNN is computationally expensive for large datasets as distances need to be calculated for all points.  
- It performs poorly in high-dimensional data due to the **curse of dimensionality**.  
- Sensitive to noisy data, outliers, and feature scaling, which can distort distance calculations.

---

## **Reproducibility**  

Follow these steps to reproduce the results:
1. **Dependencies**: Ensure the following libraries are installed:  
   ```bash
   pip install pandas numpy scikit-learn matplotlib