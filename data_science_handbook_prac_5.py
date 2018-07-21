# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:49:15 2018

@author: Daniel
"""

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

# PCA
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2,200)).T
X
plt.scatter(X[:,0], X[:,1])
plt.axis('equal')

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)
print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops =dict(arrowstyle='->',
                     linewidth=2,
                     shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    
# data plotting
plt.scatter(X[:,0], X[:, 1], alpha=.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector*3*np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal');


pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print('original shape: ', X.shape)
print('transformed shape: ', X_pca.shape)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:,0], X[:,1], alpha=.2)
plt.scatter(X_new[:,0], X_new[:,1], alpha=.8)
plt.axis('equal')

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA(2)
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:,0], projected[:,1],
            c=digits.target, edgecolor='none', alpha=.5,
            cmap = plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('compoent 2')
plt.colorbar();


pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10,4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=.1, wspace=.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),
                  cmap='binary', interpolation='nearest', clim=(0,16))

plot_digits(digits.data)

np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)

pca = PCA(.5).fit(noisy)
pca.n_components_

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(150)
pca.fit(faces.data)



























