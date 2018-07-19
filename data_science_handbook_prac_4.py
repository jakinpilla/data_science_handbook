# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:40:04 2018

@author: dsc
"""

# bias, variance, underfitting, overfitting

from os import getcwd, chdir
getcwd()
# chdir('C:/Users/dsc/data_science_handbook')
chdir('C:/Users/daniel/data_science_handbook')

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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

import numpy as np
def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1)**2
    y = 10-1./(X.ravel() + .1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)

X.ravel()
y

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
X_test = np.linspace(-.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))

plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')


from sklearn.learning_curve import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree', degree, cv=7)

plt.plot(degree, np.median(train_score, 1), color='blue',label='training score')
plt.plot(degree, np.median(val_score, 1), color='red',label='validation score')
plt.legend(loc='best') # loc = 'best'!!
plt.ylim(0,1)
plt.xlabel('degree')
plt.ylabel('score')

plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test);
plt.axis(lim);

X2, y2 = make_data(200)
plt.scatter(X2.ravel(), y2);

degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2, 
                                            'polynomialfeatures__degree', degree, cv=7)
plt.plot(degree, np.median(train_score2, 1), color='blue', label='traiaing score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=.3, linestyle='dashed')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=.3, linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')

from sklearn.learning_curve import learning_curve

fig, ax = plt.subplots(1,2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=.95, wspace=.1)

for i, degree in enumerate([2 ,9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree), X, y, cv=7, 
                                         train_sizes=np.linspace(.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', 
      linestyle='dashed')
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree= {0}'.format(degree), size=14)
    ax[i].legend(loc='best')

# GridSearchCV meta-estimator

from sklearn.model_selection import GridSearchCV

param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)

grid.fit(X, y);

grid.best_params_

model = grid.best_estimator_
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test, hold=True)
plt.axis(lim);

# feature engineering

# categorical data, derived feature, vectorization

data = [
        {'price' : 850000, 'rooms' : 4, 'nieghborhood' : 'Queen Anne'},
        {'price' : 700000, 'rooms' : 3, 'nieghborhood' : 'Fremont'},
        {'price' : 650000, 'rooms' : 3, 'nieghborhood' : 'Wallingford'},
        {'price' : 600000, 'rooms' : 2, 'nieghborhood' : 'Fremont'},
        ]

# one-hot encoding
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)
vec.get_feature_names()
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)

sample = ['problem of evil',  'evil queen', 'horizon problem']

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
X

import pandas as pd
pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);

from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)

model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)



