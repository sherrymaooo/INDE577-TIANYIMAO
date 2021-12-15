# Supervised Learning
## Table Content
* [Introduction](#Introduction)
* [How unsupervised learning works](#unsupervised)
* [Supervised learning algorithms](#Algorithms)
    - PCA(#Algorithms)
    - K-Means Clustering(#Algorithms)
* [Challenges of supervised learning](#Challenges)
* [Reference](#Reference)

---
<a class="anchor" id="Introduction"></a>
## Introduction   
**Unsupervised learning** is a type of machine learning in which the algorithm is not provided with any pre-assigned labels or scores for the training data. As a result, unsupervised learning algorithms must first self-discover any naturally occurring patterns in that training data set. Common examples include clustering, where the algorithm automatically groups its training examples into categories with similar features, and principal component analysis, where the algorithm finds ways to compress the training data set by identifying which features are most useful for discriminating between different training examples, and discarding the rest. This contrasts with supervised learning in which the training data include pre-assigned category labels (often by a human, or from the output of non-learning classification algorithm). Other intermediate levels in the supervision spectrum include reinforcement learning, where only numerical scores are available for each training example instead of detailed tags, and semi-supervised learning where only a portion of the training data have been tagged.

---
<a class="anchor" id="unsupervised"></a>
## How supervised learning works 
Unsupervised learning models are utilized for three main tasks—clustering, association, and dimensionality reduction. Clustering is a data mining technique which groups unlabeled data based on their similarities or differences. Clustering algorithms are used to process raw, unclassified data objects into groups represented by structures or patterns in the information. Clustering algorithms can be categorized into a few types, specifically exclusive, overlapping, hierarchical, and probabilistic.
Here we only introduce the PCA and K-Means Clustering algorithms.

---
<a class="anchor" id="Algorithms"></a>
##Supervised learning algorithms
* PCA 
Principal Component Analysis (PCA), is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.
* K-Means Clustering
K-Means Clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

---
<a class="anchor" id="Challenges"></a>
## Challenges of unsupervised learning
While unsupervised learning has many benefits, some challenges can occur when it allows machine learning models to execute without any human intervention. Some of these challenges can include:

* Computational complexity due to a high volume of training data
* Longer training times
* Higher risk of inaccurate results
* Human intervention to validate output variables
* Lack of transparency into the basis on which data was clustered

---
<a class="anchor" id="Reference"></a>
## Reference
1.Wikimedia Foundation. (2021, December 1). Unsupervised learning. Wikipedia. Retrieved December 13, 2021, from https://en.wikipedia.org/wiki/Unsupervised_learning. 
2.IBM Cloud Education. (n.d.). What is unsupervised learning? IBM. Retrieved December 13, 2021, from https://www.ibm.com/cloud/learn/unsupervised-learning. 