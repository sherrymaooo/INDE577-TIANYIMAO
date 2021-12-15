# Supervised Learning
## Table Content
* [Introduction](#Introduction)
* [How supervised learning works](#supervised)
    - [Classification](#supervised)
    - [Regression](#supervised)
* [Supervised learning algorithms](#Algorithms)
    - [Decision tree](#Decision)
    - [Ensemble Learning and Random Forest](#Ensemble)
    - [K-nearest neighbor](#neighbor)
    - [Linear regression](#Linear)
    - [Logistic regression](#Logistic)
    - [Multiple Neural networks](#Multiple)
    - [Perceptron](#Perceptron)
* [Challenges of supervised learning](#Challenges)
* [Reference](#Reference)

---
<a class="anchor" id="Introduction"></a>
## Introduction 
**Supervised learning (SL)** is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. 

---
<a class="anchor" id="supervised"></a>
## How supervised learning works 

Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.

Supervised learning can be separated into two types of problems when data mining—classification and regression:

* **Classification** uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.
* **Regression** is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms.

---
<a class="anchor" id="Algorithms"></a>
##Supervised learning algorithms

Various algorithms and computation techniques are used in supervised machine learning processes. Below are brief explanations of some of the most commonly used learning methods, typically calculated through use of programs like R or Python:
<a class="anchor" id="Decision"></a>
### Decision tree
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.  
<a class="anchor" id="Ensemble"></a>
### Ensemble Learning and Random Forest  
**Ensemble learrning** is a group of predictors, and it makes decision by majority vote for classification (hard voting classifier) and averaging for regression. It's a kind of Supervised learning, which uses multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Unlike a statistical ensemble in statistical mechanics, which is usually infinite, a machine learning ensemble consists of only a concrete finite set of alternative models, but typically allows for much more flexible structure to exist among those alternatives.  
**Random Forest** is perhaps the most popular ensemble learning method for classification and regression. It constructs a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random forest correct for decision trees habit of overfitting to their training set. Random forest generally outperforms decision trees.  
<a class="anchor" id="neighbor"></a>
### K-nearest neighbor
K-nearest neighbor, also known as the KNN algorithm, is a non-parametric algorithm that classifies data points based on their proximity and association to other available data. This algorithm assumes that similar data points can be found near each other. As a result, it seeks to calculate the distance between data points, usually through Euclidean distance, and then it assigns a category based on the most frequent category or average.  
Its ease of use and low calculation time make it a preferred algorithm by data scientists, but as the test dataset grows, the processing time lengthens, making it less appealing for classification tasks. KNN is typically used for recommendation engines and image recognition.  
<a class="anchor" id="Linear"></a>
### Linear regression
Linear regression is used to identify the relationship between a dependent variable and one or more independent variables and is typically leveraged to make predictions about future outcomes. When there is only one independent variable and one dependent variable, it is known as simple linear regression. As the number of independent variables increases, it is referred to as multiple linear regression. For each type of linear regression, it seeks to plot a line of best fit, which is calculated through the method of least squares. However, unlike other regression models, this line is straight when plotted on a graph.
<a class="anchor" id="Logistic"></a>
### Logistic regression
While linear regression is leveraged when dependent variables are continuous, logistical regression is selected when the dependent variable is categorical, meaning they have binary outputs, such as "true" and "false" or "yes" and "no." While both regression models seek to understand relationships between data inputs, logistic regression is mainly used to solve binary classification problems, such as spam identification.
<a class="anchor" id="Multiple"></a>
### Multiple Neural networks
Primarily leveraged for deep learning algorithms, neural networks process training data by mimicking the interconnectivity of the human brain through layers of nodes. Each node is made up of inputs, weights, a bias (or threshold), and an output. If that output value exceeds a given threshold, it “fires” or activates the node, passing data to the next layer in the network. Neural networks learn this mapping function through supervised learning, adjusting based on the loss function through the process of gradient descent. When the cost function is at or near zero, we can be confident in the model’s accuracy to yield the correct answer.A multilayer neural network is multilayer of perceptrons. It consists an input layer, several hidden layers, output layer, fully connected weights and non-linear activation function. 
<a class="anchor" id="Perceptron"></a>
### Perceptron
A Perceptron is an algorithm for supervised learning of binary classifiers. It's a simplified model of a biological neuron, and it is a type of linear classifier, i.e., a classifier that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

---
<a class="anchor" id="Challenges"></a>
## Challenges of supervised learning
Although supervised learning can offer businesses advantages, such as deep data insights and improved automation, there are some challenges when building sustainable supervised learning models. The following are some of these challenges:

* Supervised learning models can require certain levels of expertise to structure accurately.
* Training supervised learning models can be very time intensive.
* Datasets can have a higher likelihood of human error, resulting in algorithms learning incorrectly.
* Unlike unsupervised learning models, supervised learning cannot cluster or classify data on its own.

---
<a class="anchor" id="Reference"></a>
## Reference
1.Wikimedia Foundation. (2021, November 2). Supervised learning. Wikipedia. Retrieved December 13, 2021, from https://en.wikipedia.org/wiki/Supervised_learning. 
2.IBM Cloud Education. (n.d.). What is supervised learning? IBM. Retrieved December 13, 2021, from https://www.ibm.com/cloud/learn/supervised-learning. 