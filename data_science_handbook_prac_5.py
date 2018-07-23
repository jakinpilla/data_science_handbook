# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:49:15 2018

@author: Daniel
"""

from os import getcwd, chdir
getcwd()
chdir('C:/Users/dsc/data_science_handbook')
# chdir('C:/Users/daniel/data_science_handbook')

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

from sklearn.manifold import Isomap
# select 1/4 of '1' images
data = mnist['data'][mnist['target'] == 1][::4]

fig, ax = plt.subplots(figsize=(10,10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
plot_components(data, model, images=data.reshape((-1,28,28)), ax=ax, thumb_frac=.05, 
                cmap='gray_r')

# k-means
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=.60, random_state=0)
plt.scatter(X[:,0], X[:,1], s=50);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', alpha=.5); 

from sklearn.metrics import pairwise_distances_argmin
X
X.shape
X.shape[0]
rng = np.random.RandomState(2)
rng.permutation(X.shape[0])[:4]
centers = X[rng.permutation(X.shape[0])[:4]]
labels = pairwise_distances_argmin(X, centers)


def find_clusters(X, n_clusters, rseed=2):
    # put centers randomly
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # assign labels 
        labels = pairwise_distances_argmin(X, centers)
        
        # new_centers
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        
        # 
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
    return centers, labels

centers, labels = find_clusters(X, 4)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis');

center, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')

labels = KMeans(6, random_state=0).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis');

# DBSCAN, mean-shift, affinity propagation...

from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)

labels =KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')

labels = model.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

labels.shape

from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

# t-SNE, t-distributed stochastic neighbor embedding
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

accuracy_score(digits.target, labels)

from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china);

china.shape # (height, width, RGB)

data = china / 255.0
data = data.reshape(427*640, 3)
data.shape
data[0]

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0,1), ylim=(0, 1))
    
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(title, size=20);

plot_pixels(data, title='Input color space : 16 million possible colors')

data

from sklearn.cluster import MiniBatchKMeans
kmeans= MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors')

china_reordered = new_colors.reshape(china.shape)
fig, ax = plt.subplots(1,2, figsize=(16, 6),
                       subplot_kw = dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_reordered)
ax[1].set_title('16-color Image', size=16);

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=.60, random_state=0)
X = X[:, ::-1]

from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:,0],X[:,1], c=labels, s=40, cmap='viridis');

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', zorder=2)
    
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=.5, zorder=1))
                                

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2,2))
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

from sklearn.mixture import GMM
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis');
probs = gmm.predict_proba(X)
print(probs[:5].round(3))

size = 50*probs.max(1)**2
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=size);

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    '''draw ellipse with location and covar'''
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
        width, height = 2*np.sqrt(s)
    else: 
        angle=0
        width, height = 2*np.sqrt(covariance)
        
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig*width, nsig*height, angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax= ax or plt.gca()
    labels  = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:,0], X[:,1], c=labels, s=40, cmap='viridis', zorder=2)
        
    else:
        ax.scatter(X[:,0], X[:1], s=40, zroder=2)
        
    ax.axis('equal')
    w_factor= .2/gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w*w_factor)
        
gmm = GMM(n_components=4, random_state=42)
plot_gmm(gmm, X)

X_stretched = np.dot(X, rng.randn(2,2))
X_stretched

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=size)
plt.scatter(X_stretched[:,0], X_stretched[:,1], c=labels, cmap='viridis', s=size)

gmm =  GMM(n_components=4, covarianve_type='full', random_state=42)
plot_gmm(gmm, X_stretched)

from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:,0], Xmoon[:,1])

gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)
plot_gmm(gmm2, Xmoon)

gmm16 = GMM(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon)

Xnew = gmm16.sample(400, random_state=42)
plt.scatter(Xnew[:,0], Xnew[:,1])

# AIC :: Akaike information criterion
# BIC :: Baysian information criterion

n_components = np.arange(1,21)
models = [GMM(n, covariance_type='full', random_state=0).fit(Xmoon) for n in n_components]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8), 
                           subplot_kw = dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0,16)

plot_digits(digits.data)

from sklearn.ddecomposition import PCA
pca = PCA(.99, whiten=True)
data = pca.fit_transform(digits.data)
data.shape

n_components = np.arange(50,210,10)
models = [GMM(n, covariance_type='full', random_state=-0) for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics)

gmm =GMM(110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

data_new = gmm.sample(100, random_state=0)
data_new.shape

digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f*N):] += 5
    return x

x = make_data(1000)

hist =plt.hist(x, bins=30, normed=True)
hist
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
(density * widths).sum()

x = make_data(20)
bins = np.linspace(-5, 10, 10)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), 
                       sharex=True, sharey = True,
                       subplot_kw = {'xlim': (-4, 9),
                                     'ylim' : (-.02, .3)})
fig.subplots_adjust(wspace=.05)
for i, offset in enumerate([.0, .6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -.01), '|k', markeredgewidth=1)

fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -.1), '|k', markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1, alpha=.5))
        
ax.set_xlim(-4, 8)
ax.set_ylim(-.2, 8)

x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < .5) for xi in x)

plt.fill_between(x_d, density, alpha=.5)
plt.plot(x, np.full_like(x, -.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -.2, 8])


from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=.5)
plt.plot(x, np.full_like(x,-.1), '|k', markeredgewidth=1)
plt.axis([-4, 8, -.2, 5])

# atol :: absolute tolerance
# rtol :: relatice tolerance

from sklearn.neighbors import KernelDensity

kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

logprob = kde.score_samples(x_d[:,None])

plt.fill_between(x_d, np.exp(logprob), alpha=.5)
plt.plot(x, np.full_like(x, -.01), '|k', markeredgewidth=1)
plt.ylim(-.02, .22)

from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth' : bandwidths}, cv = LeaveOneOut(len(x)))
grid.fit(x[:, None])

grid.best_params_

from sklearn.base import BaseEstimator, ClassifierMixin

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """KDE based bayes generative classification 
    params
    bandwidth : kernel bandwidth
    kernel : kernel name string delivered to kernelDensity"""
    
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi) for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self
    def predict_proba(self, X):
        logprobs = np.vstack([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

from sklearn.datasets import load_digits
from sklearn.grid_search import GridSearchCV

digits = load_digits()

bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth' : bandwidths})
grid.fit(digits.data, digits.target)

scores =[val.mean_validation_score for val in grid.grid_scores_]

plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')
print(grid.best_params_)
print('accuracy =', grid.best_score_)

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
cross_val_score(GaussianNB(), digits.data, digits.target).mean()

from skimage import data, color, feature
import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualise=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')

ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features');

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape

from skimage import data, transform
imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]

from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale) for im in images for scale in [.5, 1.0, 2.0]])
    
negative_patches.shape

fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500*i], cmap='gray')
    axi.axis('off')
    
from itertools import chain
X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1
X_train.shape


