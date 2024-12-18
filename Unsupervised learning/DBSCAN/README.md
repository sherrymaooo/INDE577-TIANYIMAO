# **DBSCAN Clustering on Country Dataset**

## **Project Overview**

This project applies **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to analyze and group countries based on their socio-economic indicators. DBSCAN identifies clusters of varying shapes and density while effectively handling outliers. The goal is to understand country groupings and identify regions facing economic challenges or achieving strong performance.

---

## **Goals of the Project**

1. Implement **DBSCAN clustering** on the country dataset to identify meaningful groupings.
2. Detect **noise points** (outliers) that do not fit into any cluster.
3. Compare the results with **K-means clustering** to highlight differences in clustering structure.
4. Analyze and interpret cluster characteristics to gain socio-economic insights.

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

1. Data Preparation
2. Apply DBSCAN
3. Visualize DBSCAN Results
4. Analyze Cluster Characteristics
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
