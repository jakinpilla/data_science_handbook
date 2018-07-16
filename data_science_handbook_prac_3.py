# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:02:33 2018

@author: dsc
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

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

x = np.linspace(0,10,100)
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')

fig.savefig('./image/my_figure.png')

from IPython.display import Image
Image('./image/my_figure.png')
fig.canvas.get_supported_filetypes()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

# plt.gcf() :: get current figure
# plt.gca() :: get current axes

# object oriented
fig, ax = plt.subplots(2)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0,10,1000)
ax.plot(x, np.sin(x));

plt.plot(x, np.sin(x)) # pylab interface
plt.plot(x, np.cos(x));

plt.plot(x, np.sin(x-0), color='blue')
plt.plot(x, np.sin(x-1), color='g') # rgbcmyk
plt.plot(x, np.sin(x-2), color='0.75') # 0~1 gray scale
plt.plot(x, np.sin(x-3), color='#FFDD44')
plt.plot(x, np.sin(x-4), color=(1.0, 0.2, 0.3)) # RGB tuple
plt.plot(x, np.sin(x-5), color='chartreuse')

plt.plot(x, x+0, linestyle='solid')
plt.plot(x, x+1, linestyle='dashed')
plt.plot(x, x+2, linestyle='dashdot')
plt.plot(x, x+3, linestyle='dotted')

plt.plot(x, x+4, linestyle='-') # solid
plt.plot(x, x+5, linestyle='--') # dashed
plt.plot(x, x+6, linestyle='-.') # dashdot
plt.plot(x, x+7, linestyle=':') # dotted

plt.plot(x, x+0, '-g') # solid green
plt.plot(x, x+1, '--c') # dashed cyan
plt.plot(x, x+2, '-.k') # dashdot black
plt.plot(x, x+3, ':r') # dotted red

# xlim(), ylim()
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1,2);

plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);

plt.plot(x, np.sin(x))
plt.axis('tight')

plt.plot(x, np.sin(x))
plt.axis('equal');

plt.plot(x, np.sin(x))
plt.title('A Single Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')

# plt.xlabel() --> ax.set_xlabel()
# plt.ylabel() --> ax.set_ylabel()
# plt.xlim() --> ax.set_xlim()
# plt.ylim() --> ax.set_ylim()
# plt.title() --> ax.set_title()

ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0,10), ylim=(-2, 2), 
       xlabel= 'x', ylabel='sin(x)', 
       title='A Sample Plot')

# scatter plot
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')

rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'o', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0, 1.8)

plt.plot(x, y, '-ok')

plt.plot(x, y, '-p', color='gray', 
         markersize=15, linewidth=4, 
         markerfacecolor='white', markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2)

plt.scatter(x, y, marker='o')

rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000*rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=.3, cmap='viridis')
plt.colorbar()

from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T
plt.scatter(features[0], features[1], alpha=.2, s=100*features[3], 
            c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);

x = np.linspace(0,10,50)
dy=0.8
y = np.sin(x) + dy*np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k')
plt.errorbar(x, y, yerr=dy, fmt='o', color='black', 
             ecolor='lightgray', elinewidth=3, capsize=0);

from sklearn.gaussian_process import GaussianProcess
model = lambda x : x*np.sin(x)
xdata = np.array([1,3,5,6,8])
ydata = model(xdata)

gp=GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1, 
                   random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)
xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2*np.sqrt(MSE)

plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(xfit, yfit - dyfit, yfit+dyfit, color='gray', alpha=.2)
plt.xlim(0,10);

def f(x,y):
    return np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, colors='black')

plt.contour(X, Y, Z, 20, cmap='RdGy');

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image');

contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=.5)
plt.colorbar();

data= np.random.randn(1000)
plt.hist(data);
plt.hist(data, bins=30, normed=True, alpha=.5, histtype='stepfilled', 
         color='steelblue', edgecolor='none')
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=.3, normed=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

counts, bin_edges = np.histogram(data, bins=5)
print(counts)

mean = [0,0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
x
y
plt.hist2d(x, y, bins=30, cmap='Blues')
counts, xedges, yedges=np.histogram2d(x, y, bins=30)
counts

plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')

from scipy.stats import gaussian_kde
data = np.vstack([x, y])
kde = gaussian_kde(data)

xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6], 
           cmap = 'Blues')
cb = plt.colorbar()
cb.set_label('density')

plt.style.use('classic')
x = np.linspace(0,10,1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend();

ax.legend(loc='upper left', frameon=False)
fig

ax.legend(frameon=False, loc='lower center', ncol=2)
fig

ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig

x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5)
y = x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5)
lines = plt.plot(x, y)
plt.legend(lines[:2], ['first', 'second'])

x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5)
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)
plt.legend(lines[:2], ['first', 'second'])

plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True)

cities = pd.read_csv('./data/california_cities.csv')
cities.head()
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

plt.scatter(lon, lat, label=None, c = np.log10(population), cmap='viridis',
            s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=.3, s=area,
                label=str(area) + ' km$^2$')

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
plt.title('California Cities: Area and Population')

fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0,10,1000)
for i in range(4):
    lines += ax.plot(x, np.sin(x - i*np.pi/2), styles[i], color='black')
ax.axis('equal')
# first legend
ax.legend(lines[:2], ['line A', 'line B'], loc='upper right', frameon=False)
# second legend
from matplotlib.legend import Legend
leg =Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right', frameon=False)
ax.add_artist(leg);

x = np.linspace(0, 10, 1000)
I = np.sin(x)*np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar();
plt.imshow(I, cmap='gray')

from matplotlib.colors import LinearSegmentedColormap

def grayscale_cmap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return LinearSegmentedColormap.from_list(cmap.name + '_gray', colors, cmap.N)

def view_colormap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors =cmap(np.arange(cmap.N))
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    fig, ax = plt.subplots(2, figsize=(6, 2), 
                          subplot_kw = dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0,10,0,1])
    ax[1].imshow([grayscale], extent=[0,10,0,1])


view_colormap('jet')
view_colormap('viridis')
view_colormap('cubehelix')
view_colormap('RdBu')


