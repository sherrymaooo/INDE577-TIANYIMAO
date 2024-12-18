# Student Depression Prediction Using Decision Tree

This repository demonstrates the implementation of the **Decision Tree** algorithm to predict depression status among students. Decision Trees are intuitive, interpretable, and powerful machine learning models that split data into branches based on feature conditions to make predictions.

---

## File Descriptions  

The notebook `Decision_Tree.ipynb` contains the step-by-step implementation of the Decision Tree model for predicting depression status. It follows a structured workflow that ensures reproducibility and clear understanding.

### Workflow Outline  
This project follows a standard four-step machine learning process: 
## Part I: Student Depression Predicting using Classification 
- **Step 1**: Data Cleaning and Analysis  
- **Step 2**: Train the Model  
- **Step 3**: Test the Model  
- **Step 4**: Compare the Results  

# Part II: Student GPT Prediction using Regression

- **Step 1**: Data Cleaning and Analysis  
- **Step 2**: Train the Model  
- **Step 3**: Evaluate the Model  
- **Step 4**: Interpret the Results

---

## Knowledge About Decision Tree  

A **Decision Tree** is a supervised machine learning model that uses a tree-like structure to make predictions. Each internal node represents a condition based on a feature, and the branches represent possible outcomes. The process continues until reaching a leaf node, which provides the final prediction.

### How It Works:  
1. The model splits the dataset recursively using feature thresholds that minimize impurity.  
2. Impurity measures like **Gini Index** or **Entropy** are used to determine the optimal split.  
3. Decision Trees can be used for both **classification** (categorical target) and **regression** (numerical target) tasks.  

### Advantages of Decision Trees:  
- Simple to interpret and visualize.  
- Requires minimal data preprocessing.  
- Handles both numerical and categorical features.  

### Disadvantages of Decision Trees:  
- Prone to **overfitting** on complex datasets.  
- Sensitive to noise and small variations in the data.  
- May require pruning or ensemble methods (e.g., Random Forests) for better generalization.

---

## Dataset Used in Applications  

The **Student Depression Dataset** contains information on mental health, academic performance, and lifestyle factors of students. The target variable is **Depression**:  
- `0`: Not Depressed  
- `1`: Depressed  

### Dataset Summary  
- **Total Samples**: 27,901  
- **Number of Features**: 18  
- **Target Variable**: Depression  

### Key Features:  
- **Age**: Numerical feature representing students' ages.  
- **CGPA**: Cumulative Grade Point Average.  
- **Work Pressure**: Self-reported level of work pressure.  
- **Study Satisfaction**: Self-reported study satisfaction score.  
- **Sleep Duration**: Hours of sleep (categorical).  
- **Dietary Habits**: Self-reported eating patterns.  
- **Family History of Mental Illness**: Binary variable indicating mental illness in family history.  
- **Suicidal Thoughts**: Binary variable for self-reported suicidal ideation.

---

## Goals and Workflow  

The primary goal of this project is to predict students' depression status using the Decision Tree algorithm. The structured steps for implementation are as follows:  

### **Step 1: Data Cleaning and Analysis**  
- Load the dataset and perform an initial exploration.  
- Handle missing values by imputing the median for numerical features.  
- Encode categorical features using **Label Encoding** to ensure compatibility with the Decision Tree model.  
- Split the data into **training** and **testing** sets (60/40 split).  

### **Step 2: Train the Model**  
- Train the Decision Tree model using the **Gini Index** to measure splits.  
- Set hyperparameters such as `max_depth` to prevent overfitting.  
- Fit the model on the training data.

### **Step 3: Test the Model**  
- Evaluate the model on the test set by predicting depression status.  
- Measure performance using metrics like:  
   - **Accuracy**: The proportion of correctly predicted instances.  
   - **Confusion Matrix**: To visualize correct and incorrect predictions.  
   - **Classification Report**: Providing precision, recall, and F1-score for both classes.  

### **Step 4: Compare the Results**  
- Summarize the model's performance metrics.  
- Interpret the results, including precision, recall, and overall accuracy.  
- Visualize the **Decision Tree structure** for insights into feature importance.  

---

## Reproducibility  

To reproduce the results, follow these steps:

### Prerequisites  
Ensure you have Python and the following libraries installed:  
- pandas
- seaborn
- numpy
- matplotlib
- scikit-learn


