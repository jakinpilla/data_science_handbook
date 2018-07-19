# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:40:04 2018

@author: dsc
"""

# bias, variance, underfitting, overfitting

from os import getcwd, chdir
getcwd()
chdir('C:/Users/dsc/data_science_handbook')
#chdir('C:/Users/daniel/data_science_handbook')

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


iris = sns.load_dataset('iris')
iris.head()
sns.pairplot(iris, hue='species', size=1.5)

X_iris = iris.drop('species', axis=1)
X_iris
X_iris.shape

y_iris = iris['species']
y_iris.shape

# linear regression
rng = np.random.RandomState(42)
x = 10*rng.rand(50)
y = 2*x - 1 + rng.randn(50)
plt.scatter(x, y)

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model

# model.fit(x, y)
# Reshape your data either using array.reshape(-1, 1) if your data has a single 
# feature or array.reshape(1, -1) if it contains a single sample.

x.reshape(-1, 1)
x.shape # data number :: 50, data_feature :: 1
x.reshape(1, -1)
x.shape ## data_number :: 1, data_feature :: 50

X = x.reshape(-1, 1)  # or X = x[:, np.newaxis]
model.fit(X, y)
model.coef_
model.intercept_

np.linspace(-1, 11).shape
xfit = np.linspace(-1,11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris)

model = GaussianNB()
model.fit(X_train, y_train)
model_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, model_pred)

# convert 2D from 4D
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)
X_2D

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
iris.head()
sns.lmplot('PCA1', 'PCA2', hue='species', data=iris, fit_reg=False)

from sklearn.mixture import GMM
model=GMM(n_components=3, covariance_type='full')
model
model.fit(X_iris)
y_gmm = model.predict(X_iris)

iris['cluster'] = y_gmm
sns.lmplot('PCA1', 'PCA2', data=iris, hue='species', col='cluster', fit_reg=False)

from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape

import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, 10, figsize=(8, 8), 
                       subplot_kw = {'xticks' : [], 'yticks':[]}, 
                       gridspec_kw = dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')

# [n_samples, n_features]

X = digits.data
X.shape
X[0]
X

y=digits.target
y.shape
y[0]
y
# Isomap :: manifold learning

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)

data_projected= iso.transform(digits.data)
data_projected.shape
data_projected

plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor = 'none', alpha=.5, cmap=plt.cm.get_cmap('viridis', 10))
plt.colorbar(label='digits label', ticks=range(10))
plt.clim(-.5, 9.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')


import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, 10, figsize=(8, 8), 
                       subplot_kw = {'xticks' : [], 'yticks':[]}, 
                       gridspec_kw = dict(hspace=0.1, wspace=0.1))
                         
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(.05, .05, str(y_pred[i]), transform = ax.transAxes, 
            color='green' if (y_test[i] == y_pred[i]) else 'red')

# import, instance, fit, predict

from sklearn.datasets import load_iris 
iris = load_iris()    
X = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
y_model = model.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_model)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.5)

model.fit(X_train, y_train)

y2_model = model.predict(X_test)
accuracy_score(y_test, y2_model)

































