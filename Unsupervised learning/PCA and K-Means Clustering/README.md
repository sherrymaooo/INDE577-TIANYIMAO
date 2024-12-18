# **PCA and K-Means Clustering on Country Dataset**

## **Project Overview**

This project applies Principal Component Analysis (PCA) and K-means clustering to analyze and group countries based on their socio-economic indicators. By simplifying the dataset and identifying clusters, we uncover hidden patterns and insights about global country characteristics.

---

## **Goals of the Project**

1. Simplify high-dimensional data using PCA while retaining critical information.  
2. Identify natural groupings in the dataset using K-means clustering.  
3. Visualize the clustering results in a reduced 2D space for better interpretability.  
4. Analyze cluster characteristics to differentiate high, middle, and low-performing countries.

---

## **Dataset Description**

The project uses a **Country Dataset** containing socio-economic indicators for various countries. Below is the data dictionary:

| **Variable**    | **Description**                                                 |
|-----------------|-----------------------------------------------------------------|
| `country`       | Name of the country                                             |
| `child_mort`    | Child mortality rate (deaths per 1,000 live births)             |
| `exports`       | Exports as a percentage of GDP                                  |
| `health`        | Health expenditure as a percentage of GDP                       |
| `imports`       | Imports as a percentage of GDP                                  |
| `income`        | Net income per capita                                           |
| `inflation`     | Inflation rate (percentage)                                     |
| `life_expec`    | Life expectancy (years)                                         |
| `total_fer`     | Total fertility rate (births per woman)                         |
| `gdpp`          | GDP per capita                                                  |

---

## **Steps Covered in the Project**

1. Data Preparation and Exploration
2. Principal Component Analysis (PCA)
3. Elbow Method for K-Means
4. K-Means Clustering
5. Cluster Analysis

---

## **Reproducibility**

### **Dependencies**

To reproduce the analysis, the following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the dependencies using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn



