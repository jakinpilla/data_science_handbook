# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:02:33 2018

@author: dsc
"""

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
import matplotlib as mpl

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

x = np.linspace(0, 10, 1000)
I = np.sin(x)*np.cos(x[:, np.newaxis])
I.shape
speckles = (np.random.random(I.shape) < 0.01)
speckles.sum()
I[speckles]
I[speckles].shape
np.random.normal(0, 3, 3)
np.count_nonzero(speckles)
np.random.normal(0, 3, np.count_nonzero(speckles))
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I,cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1)

plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1);

from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
digits

digits.images[0]

fig, ax= plt.subplots(8, 8, figsize=(10, 10))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])

# manifold learning projection
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)

digits.data[0]
digits.images[0]
projection[:, 0]
projection[:, 0].shape
projection[:, 1]
projection[:, 1].shape

# plotting
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)

# plt.axes
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2))
x = np.linspace(0,10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')

fig = plt.figure()
fig.subplots_adjust(hspace=.4, wspace=.4)
for i in range(1, 7):
    ax = fig.add_subplot(2,3,i)
    ax.text(.5, .5, str((2, 3, i)), fontsize=18, ha='center')

fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')
fig

grid = plt.GridSpec(2, 3, wspace=.4, hspace=.3)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);

mean = [0, 0]
cov = [[1,1], [1,2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# gridspec axes setting
fig = plt.figure(figsize=(6, 6))
grid =plt.GridSpec(4, 4, hspace=.2, wspace=.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scattering
main_ax.plot(x, y, 'ok', markersize=3, alpha=.2)

# histogram per axis
x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
y_hist.invert_xaxis()

births = pd.read_csv('./data/births.csv')
quartiles = np.percentile(births['births'], [25,50,75])
mu, sig = quartiles[1], 0.74*(quartiles[2] - quartiles[0])
births = births.query('(births > @mu -5*@sig) & (births < @mu + 5*@sig)')

births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000*births.year + 100*births.month + births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day) for (month,day) in births_by_date.index]

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax);

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
style =dict(size=10, color='gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independences Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas", ha='right', **style)

ax.set(title='USA births by day of year (1969-1988)', ylabel='average daily births')

ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

ax.text(1,5, ". Data: (1, 5)", transform=ax.transData)
ax.text(.5,.1, ". Axes: (.5, .1)", transform=ax.transAxes)
ax.text(.2,.2, ". Figure: (.2, .2)", transform=fig.transFigure);

ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
fig

fig, ax = plt.subplots()
x = np.linspace(0,20,1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10,4), 
            arrowprops = dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5*np.pi, -1), xytext=(2, -6), 
            arrowprops = dict(arrowstyle="->", connectionstyle='angle3,angleA=0,angleB=-90'))

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data', 
            xytext=(50,-30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'))

ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data', 
            bbox=dict(boxstyle='round', fc='none', ec='gray'),
            xytext=(10,-40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle='->'))

ax.annotate("Labor Day", xy=('2012-9-4', 4850), xycoords='data', ha='center',
            xytext=(0,-20), textcoords='offset points')

ax.annotate('', xy=('2012-9-1', 4850),
            xycoords='data', textcoords='data',
            arrowprops={'arrowstyle' : '|-|,widthA=0.2,widthB=0.2',})

ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data', 
            xytext=(-80,-40), textcoords='offset points',
            arrowprops=dict(arrowstyle='fancy', 
                            fc='0.6', ec='none',
                            connectionstyle='angle3,angleA=0,angleB=-90'))
            
ax.annotate('Thanksgiving', xy=('2012-11-25', 4500), xycoords='data', 
            xytext=(-120,-60), textcoords='offset points',
            bbox=dict(boxstyle='round4,pad=.5', fc='0.9'),
            arrowprops=dict(arrowstyle='->', connectionstyle='angle,angleA=0,angleB=80,rad=20'))

ax.annotate('Christmas', xy=('2012-12-25', 3850), xycoords='data', 
            xytext=(-30,0), textcoords='offset points',
            size=13, ha='right', va='center', 
            bbox=dict(boxstlye='round', alpha=0.1),
            arrowprops=dict(arrowstyle='wedge.tail_width=0.5', alpha=.1));

ax= plt.axes(xscale='log', yscale='log')

print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())
print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())

ax = plt.axes()
ax.plot(np.random.rand(50))
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())

fig, ax = plt.subplots(5, 5, figsize=(5,5))
fig.subplots_adjust(hspace=0, wspace=0)

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10*i + j], cmap='bone')
        
fig, ax = plt.subplots(10, 10, figsize=(10,10))
fig.subplots_adjust(hspace=0, wspace=0)

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(10):
    for j in range(10):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10*i + j], cmap='bone')
        
fig, ax= plt.subplots(4,4, sharex=True, sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
fig

fig, ax = plt.subplots()
x = np.linspace(0, 3*np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3*np.pi);

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/4))
fig

def format_func(value, tick_number):
    N = int(np.round(2*value/np.pi))
    
    if N == 0:
        return '0'
    elif N == 1:
        return r'$\pi/2$'
    elif N == 2:
        return r'$\pi$'
    elif N % 2 > 0:
        return r'${0}\pi/2$'.format(N)
    else:
        return r'${0}\pi$'.format(N//2)


ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig

# rc :: runtime configuration
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np

x = np.random.randn(1000)
plt.hist(x);

ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)

plt.grid(color='w', linestyle='solid')

for spine in ax.spines.values():
    spine.set_visible(False)
    
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')

IPython_default = plt.rcParams.copy()

from matplotlib import cycler
colors = cycler('color', 
                ['#EE6666', '#3388BB', '#9988DD', 
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

plt.hist(x)

for i in range(4):
    plt.plot(np.random.rand(10))

plt.style.available[:5]

def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')

plt.rcParams.update(IPython_default);

hist_and_lines()

with plt.style.context('fivethirtyeight'):
    hist_and_lines()

with plt.style.context('ggplot'):
    hist_and_lines()

with plt.style.context('bmh'):
    hist_and_lines()

with plt.style.context('dark_background'):
    hist_and_lines()

with plt.style.context('grayscale'):
    hist_and_lines()

import seaborn
hist_and_lines()

from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

ax = plt.axes(projection='3d')
zline = np.linspace(0,15,1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
zdata= 15*np.random.random(100)
xdata = np.sin(zdata) + .1*np.random.randn(100)
ydata = np.cos(zdata) + .1*np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(60, 35)
fig

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe')

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface')

# partial polar grid
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9*np.pi, 0.8*np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r*np.sin(theta)
Y = r*np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

theta = 2*np.pi*np.random.random(1000)
r = 6*np.random.random(1000)
x = np.ravel(r*np.sin(theta))
y = np.ravel(r*np.cos(theta))
z = f(x, y)

ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=.5);

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

theta = np.linspace(0, 2*np.pi,30)
w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)

phi = 0.5*theta

r = 1 + w*np.cos(phi)
x = np.ravel(r*np.cos(theta))
y = np.ravel(r*np.sin(theta))
z = np.ravel(w*np.sin(phi))

from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))
ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z, triangles=tri.triangles, cmap='viridis', linewidths=0.2);

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# basemap is only installed in python 2.X

import matplotlib.pyplot as plt
plt.style.use('classic')

rng = np.random.RandomState(0)
rng
x = np.linspace(0,10,500)
y = np.cumsum(rng.randn(500, 6), 0)

plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')

import seaborn as sns
sns.set()

plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')

data = np.random.multivariate_normal([0,0], [[5,2], [2,2]], size=2000)
data
data = pd.DataFrame(data, columns=['x', 'y'])
for col in 'xy':
    plt.hist(data[col], normed=True, alpha=.5)

for col in 'xy':
    sns.kdeplot(data[col], shade=True)

sns.distplot(data['x'])
sns.distplot(data['y'])

sns.kdeplot(data);
with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind='kde')

with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind='hex')


iris = sns.load_dataset('iris')
iris.head()
sns.pairplot(iris, hue='species', size=2.5)

tips = sns.load_dataset('tips')
tips.head()
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
grid = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)
grid.map(plt.hist, 'tip_pct', bins=np.linspace(0,40,15))

with sns.axes_style(style='ticks'):
    g = sns.factorplot('day', 'total_bill', 'sex', data=tips, kind='box')
    g.set_axis_label('Day', 'Total Bill')

tips.head()
with sns.axes_style('white'):
    sns.jointplot('total_bill', 'tip', data=tips, kind='hex')
    
sns.jointplot('total_bill', 'tip', data=tips, kind='reg')

planets = sns.load_dataset('planets')
planets.head()

with sns.axes_style('white'):
    g = sns.factorplot('year', data=planets, aspect=2, kind='count', color='steelblue')
    g.set_xticklabels(step=5)

with sns.axes_style('white'):
    g = sns.factorplot('year', data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')

# marathon time data EDA
    
data = pd.read_csv('./data/marathon-data.csv')
data.head()
data.dtypes


from datetime import timedelta

def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s)

data = pd.read_csv('./data/marathon-data.csv', converters={'split': convert_time, 
                                                           'final' : convert_time})

data.head()
data.info()
data.dtypes

data['split_sec'] = data['split'].apply(lambda x : x.seconds)
data.head()
data['final_sec'] = data['final'].apply(lambda x : x.seconds)
data.head()

with sns.axes_style('white'):
    g = sns.jointplot('split_sec', 'final_sec', data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000), np.linspace(8000, 32000), ':k')

data['split_frac'] =1-2*data['split_sec']/data['final_sec']
data.head()
sns.distplot(data['split_frac'], kde=False)
plt.axvline(0, color='k', linestyle='--')

g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],
                 hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();

data.head()
sns.kdeplot(data[data.gender == 'M'].split_frac, label='men', shade=True)
sns.kdeplot(data[data.gender == 'W'].split_frac, label='women', shade=True)
plt.xlabel('split_frac')

sns.violinplot('gender', 'split_frac', data=data, palette = ['lightblue', 'lightpink'])

data['age_dec']= data.age.map(lambda age: 10*(age //10)) # 10 ages
data.head()

men = (data.gender == 'M')
women = (data.gender == 'W')

with sns.axes_style(style = None):
    sns.violinplot('age_dec', 'split_frac', hue='gender', data=data, 
                   split=True, inner='quartile', 
                   palette=['lightblue', 'lightpink']);

(data.age > 80).sum()

g = sns.lmplot('final_sec', 'split_frac', col='gender', data=data, 
               markers='.', scatter_kws=dict(color='c'))

g.map(plt.axhline, y=.1, color='k', ls=':')
































