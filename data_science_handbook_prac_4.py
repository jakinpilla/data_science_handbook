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

from numpy import nan
X = np.array([[nan, 0, 3],
              [3, 7, 9],
              [3, 5, 2],
              [4, nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])

from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
X2

model = LinearRegression().fit(X2, y)
model.predict(X2)

from sklearn.pipeline import make_pipeline
model= make_pipeline(Imputer(strategy='mean'), PolynomialFeatures(degree=2), 
                             LinearRegression())

model.fit(X, y)
print(y)
print(model.predict(X))

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y);

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18]*rng.rand(2000, 2)
ynew = model.predict(Xnew)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=.1)
plt.axis(lim)

y_prob = model.predict_proba(Xnew)
y_prob[-8:].round(2)

from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names

categories=['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])

train.target_names
train.target

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model= make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)

labels =model.predict(test.data)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')

def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category('sending a payload to the ISS')
predict_category('discussing islam vs atheism')
predict_category('determining the screen resolution')

rng = np.random.RandomState(1)
x = 10*rng.rand(50)
y = 2*x + rng.randn(50)
plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0,10,1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

print('Model slope :', model.coef_[0])
print('Model intercept :', model.intercept_)


rng = np.random.RandomState(1)
X = 10*rng.rand(100, 3)
X
plt.hist(X)
X.shape
X[:, 1]
plt.hist(X[:, 0], alpha=.3)
plt.hist(X[:, 1], alpha=.3)
plt.hist(X[:, 2], alpha=.3)

X[:,0].mean()
X[:,0].std()

X[:,1].mean()
X[:,1].std()

X[:,2].mean()
X[:,2].std()

y = .5 + np.dot(X, [1.5, -2, 1.])
y.shape
X.shape

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

x = np.array([2, 3, 4])
x[:, None]
x.reshape(-1, 1)
x[:, np.newaxis]

from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
poly.fit_transform(x[:, None])

from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

rng = np.random.RandomState(1)
x = 10*rng.rand(50)
x
y = np.sin(x) + .1*rng.randn(50)
y
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """1차원 입력에 대해 균일한 간격을 가지는 가우시안 특징"""
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x-y)/width
        return np.exp(-.5*np.sum(arg**2, axis))
    
    def fit(self, X, y=None):
        # 데이터 범위를 따라 펼쳐진 N개의 중앙점 생성
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, 
                                    self.width_, axis=1)
        
gauss_model= make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0,10)

model = make_pipeline(GaussianFeatures(30), LinearRegression())
model.fit(x[:, np.newaxis], y)

plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))

def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)
        
    ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0,10))

model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)

from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=.1))
basis_plot(model, title='Ridge Regression')

from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=.001))
basis_plot(model, title='Lasso Regression')

import pandas as pd
counts = pd.read_csv('./data/FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('./data/BicycleWeather.csv', index_col='DATE', 
                      parse_dates=True)

daily = counts.resample('d').sum()
daily.head()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']]
daily.head()
daily.index.dayofweek

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

daily.head()

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)
daily.head()

def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """해당 날짜의 일조시간을 계산"""
    days =(date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * 
                     np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot();
plt.ylim(8,17)

weather.head()
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp(C)'] = .5 * (weather['TMIN'] + weather['TMAX'])
weather.columns
weather['PRCP']
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily.head()
daily = daily.join(weather[['PRCP', 'Temp(C)', 'dry day']])

(daily.index - daily.index[0]).days / 365
daily['annual'] = (daily.index - daily.index[0]).days / 365
daily.head()

daily.dropna(axis=0, how='any', inplace=True)

daily.columns
X = daily.drop('Total', axis=1)
X.head()
y = daily['Total']
y.head()

model = LinearRegression(fit_intercept = False)
model.fit(X, y)
daily['predicted'] = model.predict(X)

daily[['Total', 'predicted']].plot(alpha=.3, figsize = (15, 5))

params = pd.Series(model.coef_, index=X.columns)
params

from sklearn.utils import resample
np.random.seed(1)
# bootstrapping
[model.fit(*resample(X,y)).coef_ for i in range(1000)]
err = np.std([model.fit(*resample(X,y)).coef_ for i in range(1000)], axis=0)
print(pd.DataFrame({'effect' : params.round(0), 'error' : err.round(0)}))

# SVM
# discrimination classification

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
for m, b in [(1, 0.65), (.5, 1.6), (-.2, 2.9)]:
    plt.plot(xfit, m*xfit + b, '-k')
plt.xlim(-1, 3.5)

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:,0], X[:,1], c=y, cmap='autumn')

for m, b, d in [(1, .65, .33), (.5, 1.6, .55), (-.2, 2.9, .2)]:
    yfit = m*xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit-d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=.4)

plt.xlim(-1, 3.5)

from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X,y)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """2차원 SVC를 위한 의사결정 함수 플로팅하기"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 모델 평가를 위한 그리드 생성
    x = np.linspace(xlim[0], ylim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # 의사결정 경계와 마진 플로팅
    ax.contour(X, Y, P, colors='k',
               levels=[-1,0,1], alpha=.5, linestyles=['--', '-', '--'])
    
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='blue');
                   
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);
    
model.support_vectors_

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=.6)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    plot_svc_decision_function(model, ax)
    
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=.95, wspace=.1)
for axi, N in zip(ax, [60,120]):
    plot_svm(N, axi)
    axi.set_title('N={0}'.format(N))
    
from ipywidgets import interact, fixed
interact(plot_svm, N=[10,200], ax=fixed(None));

from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)

# radial basis func
r = np.exp(-(X**2).sum(1))

from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:,0], X[:,1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    
interact(plot_3D, elev=[-90,90], azip=(-180,180),
         X = fixed(X), y=fixed(y));

# rbf :: radial basis function
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
            s=300, lw=1, facecolors='none')

# adjust svm with margin
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')

# C
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=.8)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=.0625, right=.95, wspace=.1)
for axi, C in zip(ax, [10.0, .1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],
                s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.1f}'.format(C), size=14)

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    
from sklearn.svm import SVC    
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, 
                                                    random_state=42)

from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C' : [1, 5, 10, 50],
              'svc__gamma' : [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_)

model = grid.best_estimator_

yfit = model.predict(X_test)
yfit

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62,47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == y_test[i] else 'red')
fig.subtitle('Predicted Names ; Incorrect Labels in Red', size=14);

from sklearn.metrics import classification_report
print(classification_report(y_test, yfit, target_names=faces.target_names))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, 
            xticklabels = faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='rainbow')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # scatter plot
    ax.scatter(X[:,0], X[:,1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fitting
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # plotting results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=.3, levels=np.arange(n_classes + 1) - .5,
                           cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)

visualize_classifier(DecisionTreeClassifier(), X, y)

# randomforest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=.8, random_state=1)
bag.fit(X, y)

visualize_classifier(bag, X, y)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
visualize_classifier(model, X, y);

rng = np.random.RandomState(42)
x = 10*rng.rand(200)

def model(x, sigma=.3):
    fast_oscillation = np.sin(5*x)
    slow_oscillation = np.sin(.5*x)
    noise = sigma*rng.randn(len(x))
    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, .3, fmt='o')

from sklearn.ensemble import RandomForestRegressor
forest= RandomForestRegressor(200)
forest.fit(x[:, None], y)

xfit = np.linspace(0,10,1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)

plt.figure(figsize=(15, 5))
plt.errorbar(x, y, .3, fmt='o', alpha=.5)
plt.plot(xfit, yfit, '-r')
plt.plot(xfit, ytrue, '-k', alpha=.5);

from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05, wspace=.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # image label
    ax.text(0, 7, str(digits.target[i]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_pred, y_test))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

print(accuracy_score(y_pred, y_test))































