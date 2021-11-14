# K-Nearest Neighbors

This repository represents the implementation and applications of K-Nearest Neighbors (KNN). 

## File descriptions
"K-Nearest Neighbors.ipynb" contains the describtion and my hard coding of KNN algorithm, then it's applied on a classification and a regression problem. Further the affect of *k* is explored on a dataset.

Outline:
* Algorithm describtion
* Hard code of KNN
* Application on classification problem using *wine* dataset
* Application on regression problem using *boston housing* dataset
* Impact of *k* in KNN

## Dataset used in applications
Both of the *wine* and *boston housing* dataset is loaded from *sklearn.datasets*.
* *wine* dataset 

The *wine* dataset is a classic and very easy multi-class classification dataset. It has 3 classes of target labels, each label has samples of 59, 71, 48, and the total sample size is 178 without any missing value. 

There're 13 features and there're all numeric, real, and positive. These features are Alcohol Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines and Proline.

* *boston housing* dataset

The *boston housing* dataset provides the median Boston house-price data by Harrison, D. and Rubinfeld, D.L. It has 1 numeric target labels, and the total sample size is 506 without missing value. There're 13 features and there're all numeric, real, and positive. 

