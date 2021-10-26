import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from SingleNeuron import Perceptron
from sklearn import datasets

df = datasets.load_iris(as_frame=True)
data = df['frame']

X = data[['sepal length (cm)', 'sepal width (cm)']].iloc[:100].to_numpy()
Y = data['target'].iloc[:100].to_numpy()