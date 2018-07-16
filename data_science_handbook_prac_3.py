# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:02:33 2018

@author: dsc
"""

from os import getcwd, chdir
getcwd()
chdir('C:/Users/dsc/data_science_handbook')

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

















