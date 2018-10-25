# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 08:09:18 2018

@author: dsc
"""

from os import getcwd, chdir
getcwd()

#chdir('C:/Users/dsc/data_science_handbook')

chdir('C:/Users/daniel/data_science_handbook')

chdir('C:/Users/dsc/data_science_handbook')
#chdir('C:/Users/daniel/data_science_handbook')

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
chdir('C:/Users/daniel/data_science_handbook')


chdir('C:/Users/dsc/data_science_handbook')

import math
math.sin(2)
math.cos(2)

print(In)
Out

import numpy as np
np.random.seed(0)

x1 = np.random.randint(10, size=6)
x1
x2 = np.random.randint(10, size=(3, 4))
x2
x3 = np.random.randint(10, size=(3, 4, 5))
x3

print('x3 ndim:', x3.ndim)
print('x3 shape: ', x3.shape)
print('x3 size:', x3.size)
print('dtpye:', x3.dtype)
print('itemsize:', x3.itemsize, 'bytes')
print('nbyte:', x3.nbytes, 'bytes') # nbyte = itemsize*size

x1
x1[0]
x1[4]
x1[-1]
x1[-2]
x2
x2[0,0]
x2[2,0]
x2[2,-1]
x2[0,0]=12
x2
x1[0]=3.14
x1
x=np.arange(10)
x
x[:5]
x[5:]
x[4:7]
x[::2]
x[1::2]
x[::-1]
x[5::-2]

x2
x2[:2, :3]
x2[:3, ::2]
x2[::-1, ::-1]

print(x2[:, 0])
print(x2[0, :])
print(x2[0]) # it is same with x2[0, :]

# view
print(x2)
x2_sub = x2[:2, :2]
print(x2_sub)
x2_sub[0,0] = 99
print(x2_sub)
print(x2)

x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
print(x2)

# reshape
np.arange(1, 10).reshape(3, 3)
np.arange(1, 10).reshape((3, 3))
grid = np.arange(1,10).reshape((3, 3))
print(grid)
x = np.array([1, 2, 3])
x.reshape((1, 3))
x[np.newaxis, :]
x.reshape((3, 1))
x[:, np.newaxis]

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
z = [99, 99, 99]
print(np.concatenate([x, y, z]))

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

np.concatenate([grid, grid])
np.concatenate([grid, grid], axis=1)

# vstack, np.hstack
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
np.vstack([x, grid])

y = np.array([[99],
             [99]])
np.hstack([grid, y])

# np.split, np.hsplit, np.vsplit
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
grid
upper, lower = np.vsplit(grid, [2])
upper
lower
left, right = np.hsplit(grid, [2])
left
right

# vectorized, universal functions, ufuncs
np.random.seed(0)
np.empty(3)
np.empty(5)
values = np.random.randint(1, 10, size=5)
len(values)
np.empty(len(values))

def compute_reciprocals(values):
    output = np.empty(len(values)) # room for return values
    for i in range(len(values)):
        output[i] = 1.0/values[i]
    return output

compute_reciprocals(values)
big_array = np.random.randint(1, 100, size=10000000)

print(compute_reciprocals(values))
print(1.0/values)

compute_reciprocals(big_array)
1/big_array
np.arange(5)
np.arange(1, 6)
np.arange(5) / np.arange(1, 6)
x = np.arange(9).reshape((3, 3))
2**x

x = np.arange(4)
print('x = ', x)
print('x+5=', x+5)
print('x-5=', x-5)
print('x*2=', x*2)
print('x/2=', x/2)
print('x//2=', x//2)
print('-x = ', -x)
print('x**2=', x**2)
print('x%2=', x%2)
-(0.5*x + 1)**2
np.add(x, 2)
x = np.array([-2, -1, 0, 1, 2])
abs(x)
np.absolute(x)
np.abs(x)
x = np.array([3-4j, 4-3j, 2+0j, 0+1j])
np.abs(x)

# trigonometric
theta = np.linspace(0, np.pi, 3)
print('theta = ', theta)
print('sin(theta)= ', np.sin(theta))
print('cos(theta)=', np.cos(theta))
print('tan(theta)=', np.tan(theta))

x=[-1, 0, 1]
print('x=', x)
print('arcsin(x)=', np.arcsin(x))
print('arccos(x)=', np.arccos(x))
print('arctan(x)=', np.arctan(x))

# exponential and log
x=[1, 2, 3]
print('x =', x)
print('e^x=', np.exp(x))
print('2^x=', np.exp2(x))
print('3^x=', np.power(3,x))

x=[1, 2, 4, 10]
print('x =', x)
print('ln(x)= ', np.log(x))
print('log2(x)=', np.log2(x))
print('log10(x)=', np.log10(x))

x=[0, 0.001, 0.01, 0.1]
print('exp(x)-1=', np.expm1(x))
print('log(1+x)=', np.log1p(x))

from scipy import special
x = [1, 5, 10]
print('gamma(x)=', special.gamma(x))
print('ln|gamma(x)| = ', special.gamma(x))
print('beta(x,2)=', special.beta(x, 2))

x = np.array([0, 0.3, 0.7, 1.0])
print('erf(x) =', special.erf(x))
print('erfc(x)=', special.erfc(x))
print('erfinv(x)=', special.erfinv(x))

# designate out location
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y=np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

# reduce(), accumulate()
x = np.arange(1, 6)
np.add.reduce(x)
np.multiply.reduce(x)
np.add.accumulate(x)
np.multiply.accumulate(x)

np.sum(x)
np.prod(x)
np.cumsum(x)
np.cumprod(x)

# outer products
x = np.arange(1, 6)
np.multiply.outer(x, x) # product table

L = np.random.random(100)
sum(L)
np.sum(L)

big_array = np.random.rand(10000000)
min(big_array)
max(big_array)
np.min(big_array)
np.max(big_array)
print(big_array.min(), big_array.max(), big_array.sum())
M = np.random.random((3, 4))
print(M)
M.sum()
M.min(axis=0)
M.min(axis=1)

import pandas as pd
data = pd.read_csv('./data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
heights.mean()
heights.std()
heights.min()
heights.max()
np.percentile(heights, 25)
np.median(heights)
np.percentile(heights, 75)

import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.hist(heights)
plt.title('Heights Distribution of US President')
plt.xlabel('height (cm)')
plt.ylable('number')

# broadcasting
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a+b
a+5
M = np.ones((3, 3))
M
M + a
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
a+b

# broadcast rules
# Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions 
# is padded with ones on its leading (left) side.

# Rule 2: If the shape of the two arrays does not match in any dimension, the array 
# with shape equal to 1 in that dimension is stretched to match the other shape.

# Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

M = np.ones((2, 3))
a = np.arange(3)

M.shape 
a.shape
M + a

a = np.arange(3).reshape((3, 1))
b = np.arange(3)
a.shape
b.shape
a+b

M = np.ones((3, 2))
a = np.arange(3)
M.shape
a.shape
# M + a error :: ValueError: operands could not be broadcast together with shapes (3,2) (3,) 

a[:, np.newaxis].shape
M + a[:, np.newaxis]
np.logaddexp(M, a[:, np.newaxis])

X = np.random.random((10, 3))
Xmean = X.mean(0)
Xmean.shape
Xmean
X_centered = X - Xmean
X_centered.mean(0)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)

import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar()

rainfall = pd.read_csv('./data/Seattle2014.csv')['PRCP'].values

rainfall = pd.read_csv('./data/Seattle2014.csv', parse_dates=['DATE'])
rainfall.head()
rainfall = rainfall[['DATE', 'PRCP']]
rainfall[['DATE', 'PRCP']].plot(x='DATE', y='PRCP', figsize=(15, 5));

inches = rainfall / 254
inches.shape
plt.hist(inches, 40);


x = np.array([1, 2, 3, 4, 5])
x < 3
x > 3
x <= 3
x >= 3
x != 3
(2*x) == (x**2)
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
x
x < 6
print(x)
np.count_nonzero(x<6)
np.sum(x < 6)
np.sum(x < 6, axis=1)
np.any(x > 8)
np.any(x < 0)
np.all(x < 10)
np.all(x == 6)
np.all(x < 8, axis=1)

np.sum((inches > 0.5) & (inches < 1))
np.sum(~((inches <=0.5) | (inches >= 1)))
print('Number days without rain : ', np.sum(inches==0))
print('Number days with rain : ', np.sum(inches !=0))
print('Days with more than 0.5 inches : ' , np.sum(inches > 0.5))
print('Rainy days with < 0.1, inches : ', np.sum((inches > 0) & (inches < 0.2)))

x
x < 5
x[x<5]
rainy = (inches > 0)
summer = (np.arange(365) -172 < 90) & (np.arange(365) -172 > 0)
print('Median precip on rainy days in 2014 (inches) : ', np.median(inches[rainy]))
print('Median precip on summmer days in 2014 (inches) : ', np.median(inches[summer])) # 0.0 why??
print('Maximum precip on summer days in 2014 (inches) : ', np.max(inches[summer]))
print('Median precip on non-summer rainy days (inches) : ', np.median(inches[rainy & ~summer]))

bool(42), bool(0)
bool(42 and 0)
bool(42 or 0)
bin(42)
bin(59)
bin(42&59)
bin(42|59)
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A|B
x = np.arange()
x = np.arange(10)
(x>4) & (x<8)

# fancy indexing
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
[x[3], x[7], x[2]]
ind = [3, 7, 4]
x[ind]
ind = np.array([[3, 7],
                [4, 5]])

x[ind]
X = np.arange(12).reshape((3, 4))
X
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

row[:, np.newaxis]
col
X[row[:, np.newaxis], col]
row[:, np.newaxis]*col
X[2, [2, 0, 1]]
X[1:, [2, 0, 1]]
mask = np.array([1, 0, 1, 0], dtype=bool)
mask
row[:, np.newaxis]
X
X[row[:, np.newaxis], mask]

mean= [0, 0]
# covariance matrix :: cov = 2
cov = [[1, 2], 
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

plt.scatter(X[:, 0], X[:, 1])
indices = np.random.choice(X.shape[0], 20, replace=False)
indices
selection = X[indices]
selection.shape
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(selection[:, 0], selection[:, 1], facecolor='red', s=50, alpha=1)
plt.gcf().set_size_inches(20, 5)

x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i]=99
print(x)
x[i] -=10
print(x)
x = np.zeros(10)
x[[0, 0]] = [4, 6]
x
i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x
x = np.zeros(10)
np.add.at(x, i, 1)
print(x)

np.random.seed(42)
x = np.random.randn(100)
bins = np.linspace(-5, 5, 20)
counts=np.zeros_like(bins)
i = np.searchsorted(bins, x)
np.add.at(counts, i, 1)
plt.plot(bins, counts, linestyle='steps')
plt.hist(x, bins, histtype='step')

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
x = np.array([2, 1, 4, 3, 5])
selection_sort(x)

x = np.array([2, 1, 4, 3, 5])
np.sort(x)
x.sort()
print(x)
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
x[i]
rand= np.random.RandomState(42)
X = rand.randint(0,10, (4, 6))
print(X)
np.sort(X, axis=0)
np.sort(X, axis=1)
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)
np.partition(X, 2, axis=1)

X = rand.rand(10, 2)
X
X.shape
plt.scatter(X[:, 0], X[:, 1], s=100)
X[:, np.newaxis, :]
X[:, np.newaxis, :].shape
X[np.newaxis, :, :]
X[np.newaxis, :, :].shape
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences
differences.shape
sq_differences = differences**2
dist_sq = sq_differences.sum(-1)
dist_sq.shape

dist_sq.diagonal()
nearest = np.argsort(dist_sq, axis=1)
print(nearest)
K=2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)
nearest_partition 
plt.scatter(X[:, 0], X[:, 1], s=100)
K=2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        plt.plot(*zip(X[j], X[i]), color='black')

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data = np.zeros(4, dtype={'names' : ('name', 'age', 'weight'),
                       'formats' : ('U10', 'i4', 'f8')})
    
# U10 :: max length 10 Unicode string
# i4 :: 4 byte int
# # f8 :: 64bit float
    
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
data['name']
data[0]
data[-1]['name']
data[data['age'] < 30]['name']

np.dtype({'names' : ('name', 'age', 'weight'),
          'formats' : ('U10', 'i4', 'f8')})
np.dtype({'names' : ('name', 'age', 'weight')})

<<<<<<< HEAD
=======

>>>>>>> 2b28a5f8f546f4c519ce6aea35239a9a5544b1cb
np.dtype({'names' : ('name', 'age', 'weight'),
         'formats' : ((np.str_, 10), int, np.float32)})
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
np.dtype('S10, i4, f8')
tp=np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X)
print(X[0])
print(X['mat'][0])

data['age']
data_rec = data.view(np.recarray)
data_rec.age
<<<<<<< HEAD

=======
>>>>>>> 2b28a5f8f546f4c519ce6aea35239a9a5544b1cb









