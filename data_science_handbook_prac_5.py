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

# print eigenfaces
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw = {'xticks' : [], 'yticks' : []},
                         gridspec_kw = dict(hspace=.1, wspace=.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62,47), cmap='bone')


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

pca = RandomizedPCA(150).fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

# result plotting
fig, ax = plt.subplots(2,10, figsize=(10, 2.5),
                       subplot_kw = {'xticks' :[], 'yticks':[]},
                       gridspec_kw=dict(hspace=.1, wspace=.1))
for i in range(10):
    ax[0,i].imshow(faces.data[i].reshape(62,47), cmap='binary_r')
    ax[1,i].imshow(projected[i].reshape(62,47), cmap='binary_r')
    
ax[0,0].set_ylabel('full-dim\ninput')
ax[1,0].set_ylabel('150-dim\nreconstruction')

# manifold learning
# MDS :: multidimensional scaling
# LLE :: locally linear embedding
# Isomap :: isometric mapping

fig, ax = plt.subplots(figsize=(4,1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.axis('off')
ax.text(.5, .4, 'HELLO', va='center', ha='center', weight='bold', size=85)
fig.savefig('./data/hello.png')
plt.close(fig)

from matplotlib.image import imread
data = imread('./data/hello.png')
data.shape # (72, 288, 4)
data[::-1, :, 0].shape # (72, 288)
data[::-1, :, 0].T.shape # (288, 72)

data = imread('./data/hello.png')[::-1, :, 0].T
data.shape # (288, 72)

rng = np.random.RandomState(42)
X = rng.rand(4*1000, 2)
X
X.shape # (4000, 2)

(X * data.shape)
(X * data.shape).shape # (4000, 2)
(X * data.shape).astype(int).shape # (4000, 2)
(X * data.shape).astype(int).T.shape # (2, 4000)
i, j = (X * data.shape).astype(int).T # (4000, 2) * (288, 72)
i.shape # (4000,)
j.shape # (4000,)
 
mask = (data[i, j] < 1)
X[mask].shape
X = X[mask]

X[:,0] *=(data.shape[0] / data.shape[1])
X = X[:1000]

def make_hello(N=1000, rseed=42):
    fig, ax = plt.subplots(figsize=(4,1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(.5, .4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('./data/hello.png')
    plt.close(fig)
    
    from matplotlib.image import imread
    data = imread('./data/hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4*N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:,0] *= (data.shape[0]/data.shape[1])
    X=X[:N]
    return X[np.argsort(X[:, 0])]
    
X = make_hello(1000)
colorize = dict(c=X[:,0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:,0], X[:,1], **colorize) 
plt.axis('equal');

X.shape

def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
          [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)

X2 = rotate(X,20) + 5
plt.scatter(X2[:,0], X2[:,1], **colorize)
plt.axis('equal')

from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
D.shape

plt.figure(figsize=(10,10))
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()

D2 = pairwise_distances(X2)
np.allclose(D, D2)

# 거리로부터 x, y 좌표를 다시 반환하려는 것이 다차원 척도법이 하려는 일
from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)

plt.figure(figsize=(10,10))
plt.scatter(out[:,0], out[:, 1], **colorize)
plt.axis('equal');

def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])

X3 = random_projection(X,3)
X3.shape

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter(X3[:,0], X3[:,1], X3[:,2], **colorize)
ax.view_init(azim=70, elev=50)

model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)

plt.figure(figsize=(10,10))
plt.scatter(out3[:,0], out3[:,1], **colorize)
plt.axis('equal')

def make_hello_s_curve(X):
    t = (X[:,0]-2)*.75*np.pi
    x = np.sin(t)
    y = X[:,1]
    z = np.sign(t) * (np.cos(t)-1)
    return np.vstack((x,y,z)).T

XS = make_hello_s_curve(X)

from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(XS[:,0], XS[:,1], XS[:,2], **colorize)

from sklearn.manifold import MDS
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)

plt.scatter(outS[:,0], outS[:,1], **colorize)
plt.axis('equal')

from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, 
                               method='modified', eigen_solver='dense')

out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:,0], out[:,1], **colorize)
ax.set_ylim(.15, -.15)

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=30)
faces.data.shape

fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')

from sklearn.decomposition import RandomizedPCA
model = RandomizedPCA(100).fit(faces.data)

plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance')

from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)

proj.shape


from matplotlib import offsetbox

def plot_components(data, model, images=None, ax=None, thumb_frac=.05, cmap='gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:,0], proj[:,1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images)**2 ,1)
            if np.min(dist) < min_dist_2:
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)

fig, ax = plt.subplots(figsize=(15, 15))
plot_components(faces.data, model=Isomap(n_components=2), 
                images=faces.images[:, ::2, ::2])


# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
# "http://mldata.org is not working"


from sklearn.datasets import fetch_mldata
try:
    mnist = fetch_mldata('MNIST original')
except Exception as ex:        
    from six.moves import urllib
    from scipy.io import loadmat
    import os

    mnist_path = os.path.join(".", "data", "mnist-original.mat")

    # download dataset from github.
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)

    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Done!")
    
    
mnist.keys()
mnist['data'].shape

fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist['data'][1250 * i].reshape(28,28), cmap='gray_r')

data = mnist['data'][::30]
target = mnist['target'][::30]

model = Isomap(n_components=2)
proj = model.fit_transform(data)

plt.figure(figsize=(15,10))
plt.scatter(proj[:,0], proj[:,1], c=target, cmap = plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-.5, 9.5)







