# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 23:22:15 2018

@author: Daniel
"""

# pandas
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

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
data.values
data.index
data[1]
data[1:3]
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data
data['b']
population_dict = {'California' : 38332521, 
                   'Texas' : 26448193,
                   'New York' : 19651127,
                   'Florida' : 19552860,
                   'Illinois' : 12882135}
population = pd.Series(population_dict)
population
population['California']
population['California' : 'Illinois']
pd.Series([2, 4, 6])
pd.Series(5, index=[100,200,300])
pd.Series({2:'a', 1:'b', 3:'c'})
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
area_dict = {'California' : 423967, 'Texas' : 695662, 'New York' : 141297, 
             'Florida' : 170312, 'Illinois' : 149995}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'population' : population, 'area' : area})
states
states.index
states.columns
states['area']
pd.DataFrame(population, columns=['population'])

data=[{'a' : i, 'b' : 2*i} for i in range(3)]
data
pd.DataFrame(data)
pd.DataFrame([{'a':1, 'b':2}, {'b':3, 'c':4}])
pd.DataFrame({'population' : population,'area' : area})
pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A
pd.DataFrame(A)
ind = pd.Index([2, 3, 5, 7, 11])
ind
ind[1]
ind[::2]
print(ind.size, ind.shape, ind.ndim, ind.dtype)

indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB
indA | indB
indA ^ indB

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data
data['b']
'a' in data
data.keys()
list(data.items())
data['e']=1.25
data 
data['a':'c'] # 명시적 인덱스
data[0:2] # 암묵적 인덱스
# 명시적 인덱스로 슬라이싱 할 때는 최종 인덱스가 슬라이스에 포함되지만,
# 암묵적 인덱스로 슬라이싱 하면 최종 인덱스가 그 슬라이스에서 제외됨.

data[(data>0.3) & (data<0.8)]
data[['a', 'c']]

data=pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data[1]
data[1:3]
data.loc[1]
data.loc[1:3]
data.iloc[1]
data.iloc[1:3]

population_dict = {'California' : 38332521, 
                   'Texas' : 26448193,
                   'New York' : 19651127,
                   'Florida' : 19552860,
                   'Illinois' : 12882135}
area_dict = {'California' : 423967, 'Texas' : 695662, 'New York' : 141297, 
             'Florida' : 170312, 'Illinois' : 149995}

data=pd.DataFrame({'area':area, 'pop':population})
data
data['area']
data.area
data.area is data['area']
data.pop is data['pop']
data.pop
data['density'] = data['pop']/data['area']
data
data.values
data.T
data.values[0]
data.iloc[:3, :2]
data.loc[:'Illinois', :'pop']
data.ix[:3, 'pop']
data.loc[data.density > 100, ['pop', 'density']]
data.iloc[0,2]=90
data
data.loc['California', 'density'] = 100
data
data['Florida':'Illinois']
data[1:3]
data[data.density > 100]

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0,10,4))
df = pd.DataFrame(rng.randint(0,10,(3,4)), columns=['A','B', 'C', 'D'])
df
np.exp(ser)
np.sin(df*np.pi/4)
area=pd.Series({'Alaska' : 1723337, 'Texas' : 695662, 'California' : 423967},
                name='area')
population = pd.Series({'California':38332521, 'Texas' : 26448193, 
                        'New Yrok':19651127}, name='population')
population/area
area.index | population.index
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A+B
A.add(B, fill_value=0)
A=pd.DataFrame(rng.randint(0,20,(2,2)), columns=list('AB'))
A
B = pd.DataFrame(rng.randint(0,10,(3,3)),columns=list('BAC'))
B
A+B
A
A.stack()
A.stack().mean()
fill = A.stack().mean()
A.add(B, fill_value=fill)
A = rng.randint(10, size=(3, 4))
A
A[0]
A-A[0]
df = pd.DataFrame(A, columns=list('QRST'))
df-df.iloc[0]
df['R']
df.subtract(df['R'], axis=0)
df.subtract(df['R'], axis=1)
halfrow=df.iloc[0, ::2]
halfrow
df-halfrow

# None
vals1 = np.array([1, None, 3, 4])
vals1
vals2 = np.array([1, np.nan, 3, 4])
vals2
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
pd.Series([1, np.nan, 2, None])
x = pd.Series(range(2), dtype=int)
x[0]=None
x
# innull(), notnull(), dropna(), fillna()
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])

df
df.dropna()
df.dropna(axis='columns')
df.dropna(axis=0)
df.dropna(axis=1)
df[3]=np.nan
df
df.dropna(axis='columns', how='all')
df.dropna(axis='rows', thresh=3)

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
# forward fill
data.fillna(method='ffill')
data.fillna(method='bfill')
df
df.fillna(method='ffill', axis=1)

# hierarchical indexing
index=[('California', 2000), ('California', 2010), ('New York', 2000),
       ('New York', 2010), ('Texas', 2000), ('Texas', 2010)]
populations=[33871648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop=pd.Series(populations, index=index)
pop
pop[('California', 2010):('Texas', 2000)]
pop[[i for i in pop.index if i[1] == 2010]]
index = pd.MultiIndex.from_tuples(index)
index
pop=pop.reindex(index)
pop[:, 2010]
pop_df = pop.unstack()
pop_df
pop_df.stack()
pop_df = pd.DataFrame({'total':pop, 
                       'under18' : [9267089, 9284094, 
                                    4687374, 4318033,
                                    5906301, 6879014]})
pop_df
f_u18 = pop_df['under18']/pop_df['total']
f_u18
f_u18.unstack()
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
pd.MultiIndex.from_tuples([('a',1), ('a',2), ('b', 1), ('b', 2)])
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
pd.MultiIndex(levels=[['a', 'b'], [1, 2]], labels=[[0,0,1,1], [0,1,0,1]])
pop.index.names=['state', 'year']
pop

# columns multi index
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns= pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], 
                                     names=['subject', 'type'])
np.random.randn(4,6)
data=np.round(np.random.randn(4,6), 1)
data
data[:, ::2] *=10
data
data += 37
data
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
health_data['Guido']

pop
pop['California', 2000]
pop['California']
pop.loc['California':'New York']
pop[:, 2000]
pop[pop>22000000]
pop[['California', 'Texas']]
health_data['Guido']
health_data['Guido']['HR']
health_data['Guido','HR']
health_data.iloc[:2, :2]
health_data.loc[:, ('Bob', 'HR')]

idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]

# multi index arange
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names=['char', 'int']
data
data=data.sort_index()
data['a':'c']
pop
pop.unstack(level=0)
pop.unstack(level=1)
pop
pop_flat = pop.reset_index(name='population')
pop_flat
pop_flat.set_index(['state', 'year'])
health_data
data_mean = health_data.mean(level='year')
data_mean
data_mean.mean(axis=1, level='type')

# concat(), append()
def make_df(cols, ind):
    data = {c : [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

make_df('ABC', range(3))

ser1 = pd.Series(['A', 'B', 'C'], index=[1,2,3])
ser2 = pd.Series(['D','E', 'F'], index=[4,5,6])
pd.concat([ser1, ser2])
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
df1
df2
pd.concat([df1, df2])
.pd.concat([df1, df2], axis=1)
df3=make_df('AB', [0,1])
df4=make_df('CD', [0,1])
df3
df4
pd.concat([df3, df4], axis=1)

x = make_df('AB', [0,1])
y = make_df('AB', [2,3])
x
y
y.index = x.index
y
pd.concat([x, y])
try:
    pd.concat([x,y], verify_integrity=True)
except ValueError as e:
       print('ValueError:', e)
       
pd.concat([x, y], ignore_index=True)
pd.concat([x, y], keys=['x', 'y'])
df5 = make_df('ABC', [1,2])
df6 = make_df('BCD', [3,4])
print(df5), print(df6), print(pd.concat([df5, df6]))
print(df5), print(df6), print(pd.concat([df5, df6], join='inner'))
print(pd.concat([df5, df6], join_axes=[df5.columns]))

df1.append(df2)

# merge(), join()
df1 = pd.DataFrame({'employee':['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group':['Accounting', 'Engineering', 'Engineering', 'HR']})

df2 = pd.DataFrame({'employee' : ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1, df2)
df3
df4 = pd.DataFrame({'group':['Accounting', 'Engineering', 'HR'],
                    'supervisor' : ['Carly', 'Guido', 'Steve']})

df3
df4
pd.merge(df3, df4)
df5 = pd.DataFrame({
        'group' : ['Accounting', 'Accounting','Engineering','Engineering','HR','HR'],
        'skills' : ['math','spreadsheets','coding','linux','spreadsheets','organization']})
df5
df1
df5
pd.merge(df1,df5)
df1
df2
pd.merge(df1,df2,on='employee')
df3 = pd.DataFrame({'name':['Bob','Jake','Lisa','Sue'],
                    'salary':[70000,80000,120000,90000]})
df3
pd.merge(df1,df3,left_on='employee',right_on='name')
pd.merge(df1,df3,left_on='employee',right_on='name').drop('name', axis=1)

df1a=df1.set_index('employee')
df2a=df2.set_index('employee')
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)
df1a
df3
pd.merge(df1a,df3,left_index=True,right_on='name')

df6=pd.DataFrame({'name':['Peter','Paul','Mary'],
                  'food':['fish','beans','bread']}, columns=['name','food'])

df7=pd.DataFrame({'name': ['Mary','Joseph'],
                  'drink':['wine','beer']},
                  columns=['name','drink'])

df6
df7
pd.merge(df6,df7)
pd.merge(df6,df7,how='inner')
pd.merge(df6,df7,how='outer')
pd.merge(df6,df7,how='left')
pd.merge(df6, df7, how='right')

df8=pd.DataFrame({'name':['Bob','Jake','Lisa','Sue'],
                  'rank':[1,2,3,4]})
df9=pd.DataFrame({'name':['Bob','Jake','Lisa','Sue'],
                  'rank':[3,1,4,2]})
df8
df9
pd.merge(df8,df9,on='name')
pd.merge(df8,df9,on='name',suffixes=['_L', '_R'])

pop=pd.read_csv('./data/state-population.csv')
area=pd.read_csv('./data/state-areas.csv')
abbrevs=pd.read_csv('./data/state-abbrevs.csv')
pop.head()
area.head()
abbrevs.head()

merged=pd.merge(pop, abbrevs, how='outer',left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1)
merged.head()
merged.isnull().any()
merged[merged['population'].isnull()].head()
merged.loc[merged['state'].isnull(), 'state/region'].unique()

merged.loc[merged['state/region']=='PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'Unite States'
merged.isnull().any()

final = pd.merge(merged, area, on='state', how='left')
final.head()
final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna(inplace=True)
final.head()
data2010 = final.query("year==2010 & ages=='total'")
data2010.head()
data2010.set_index('state', inplace=True)
density=data2010['population']/data2010['area (sq. mi)']
density.sort_values(ascending=False, inplace=True)
density.head()
density.tail()

# aggregating and grouping
import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape
planets.head()
planets.dropna().describe()

# split, apply, combine
df
#df.groupby('key')
#df.groupby('key').sum()
planets.groupby('method')['orbital_period'].median()
for (method,group) in planets.groupby('method'):
    print('{0:30s} shape={1}'.format(method, group.shape))
planets.groupby('method')['year'].describe()

# groupby() :: aggregate(), filter(), transform(), apply()
rng=np.random.RandomState(42)
df = pd.DataFrame({'key' : ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6), 
                   'data2':rng.randint(0,10,6)},
    columns=['key', 'data1', 'data2'])

df
df.groupby('key').aggregate(['min', np.median, max])
df.groupby('key').aggregate(['min', np.median, 'max'])
df.groupby('key').aggregate([min, np.median, max])
df.groupby('key').aggregate([min, np.median, 'max'])
df.groupby('key').aggregate({'data1':'min', 'data2':max})

def filter_func(x):
    return x['data2'].std() > 4

print(df)
print(df.groupby('key').std())
df.groupby('key').filter(filter_func)

df.groupby('key').transform(lambda x : x-x.mean())

def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x

df.groupby('key').apply(norm_by_data2)
L = [0,1,0,1,2,0]
df.groupby(L).sum()
df.groupby(df['key']).sum()

df2=df.set_index('key')
mapping = {'A' : 'vowel' , 'B':'consonant', 'C':'consonant'}
df2.groupby(mapping).sum()
df2.groupby(str.lower).mean()
df2.groupby([str.lower, mapping]).mean()

import seaborn as sns
planets = sns.load_dataset('planets')
decade=10*(planets['year']//10)
decade=decade.astype(str) + 's'
planets['year'] = decade
planets.head()
planets = planets.rename(columns = {'year' : 'decade'})
planets.head()
planets.groupby(['method', 'decade'])['number'].sum().unstack().fillna(0)

titanic = sns.load_dataset('titanic')
titanic.head()
titanic.groupby('sex')[['survived']].mean()
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()

titanic.pivot_table('survived', index='sex', columns='class')

titanic = sns.load_dataset('titanic')
titanic=titanic[['sex', 'age', 'class','survived', 'fare']]
titanic = titanic.dropna()
titanic.describe()
titanic.shape
titanic.isnull().sum().sum()
age = pd.cut(titanic['age'], [0,18,80])
age.value_counts()
len(age)
age.isnull().sum()
titanic['age']=age
titanic.columns
titanic.pivot_table('survived', ['sex', 'age'], 'class')

titanic = sns.load_dataset('titanic')
titanic=titanic[['sex', 'age', 'class','survived', 'fare']]
titanic = titanic.dropna()
titanic.describe()
age = pd.cut(titanic['age'], [0,18,80])
fare=pd.qcut(titanic['fare'], 2)
age
fare
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

titanic.pivot_table(index='sex', columns='class', 
                    aggfunc={'survived' : sum, 'fare' : 'mean'})

titanic.pivot_table('survived', index='sex', columns='class', margins=True)

births=pd.read_csv('./data/births.csv')
births.head()
births['decade'] = 10*(births['year']//10)
births.head()

births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum').T

sns.set()
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()

quartiles = np.percentile(births['births'],[25,50,75])
mu=quartiles[1]
sig = 0.74*(quartiles[2] - quartiles[0])
births = births.query('(births > @mu -5*@sig) & (births < @mu + 5*@sig)')
births['day'] = births['day'].astype(int)
births.head()

births.index = pd.to_datetime(10000*births.year + 100*births.month + births.day, 
                             format='%Y%m%d')
births.head()
births['dayofweek'] = births.index.dayofweek
births.head()

births.pivot_table('births', index='dayofweek', 
                   columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Thue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')

births_by_date = births.pivot_table('births', [births.index.month, births.index.day])
births_by_date.head()
# for 2012.02.29.
births_by_date.index=[pd.datetime(2012, month, day) for (month, day) \
in births_by_date.index]
births_by_date.head()
fig, ax = plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax)

x = np.array([2, 3, 5, 7, 11, 13])
x*2
# capitalize()

data=['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]

data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
names = pd.Series(data)
names
names.str.capitalize()

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam', 'Eric Idle', 
                   'Terry Jones', 'Michael Palin'])
monte.str.lower()
monte.str.startswith('T')
monte.str.split()
monte.str.extract('([A-Za-z]+)')
monte.str.findall(r'^[^AEIOU].*[^aeiou]$') # regular exp :: why? how?
monte.str[0:3]
monte.str.split().str.get(-1)
full_monte = pd.DataFrame({'name' : monte, 
                          'info' : ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
full_monte
full_monte['info'].str.get_dummies('|')

# time_series

from datetime import datetime
datetime(year=2015, month=7, day=4)

from dateutil import parser
date = parser.parse('4th of July, 2015')
date
date = parser.parse('2015-7-4')
date
date = parser.parse('2015-07-04')
date
date = parser.parse('2015/7/4')
date
date = parser.parse('4/7/2015')
date
date = parser.parse('7/4/2015')
date

date.strftime('%A')
date.strftime('%Y%m%d')
date.strftime('%Y-%m-%d')
date.strftime('%Y-%m-%d')
date.strftime('%Y/%m/%d')

date=np.array('2015-07-04', dtype=np.datetime64)
date
date + np.arange(12)
np.datetime64('2015-07-04')
np.datetime64('2015-07-04 12:00')
np.datetime64('2015-07-04 12:59:59.50', 'ns')

date=pd.to_datetime('4th of July, 2015')
date
date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')

index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', 
                          '2015-08-04'])
data = pd.Series([0,1,2,3], index=index)
data
data['2014-07-04':'2015-07-04']
data['2015']
data['2015-07']

# DatetimeIndex, PeriodIndex, TimedeltaIndex
dates = pd.to_datetime([datetime(2015,7,3), '4th of July, 2015', 
                        '2015-July-6', '07-07-2015', '20150708'])
dates
dates.to_period('D')
dates-dates[0]

# pd.date_range()
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=8)
pd.date_range('2015-07-03', periods=8, freq='H')
pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq='H')
pd.timedelta_range(0, periods=9, freq='2H30T')

from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=5, freq=BDay())

# to pandas.api.types in Pandas 0.23.0.
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data
import fix_yahoo_finance as yf
yf.pdr_override()

#To get data:
import datetime
start = datetime.datetime(2004, 1, 1)
end = datetime.datetime(2015, 8, 31)
start
end
goog = data.get_data_yahoo('goog', start, end)
goog.head()

goog = goog['Close']
goog.plot()

# resample(), asfreq()
goog.plot(alpha=.5, style='-')
goog.resample('BA').mean().plot(style=':')
goog.asfreq('BA').plot(style='--');
plt.legend(['input', 'resample', 'asfreq'], loc='upper left')

fig, ax = plt.subplots(2, sharex=True)
data = goog.iloc[:10]
data.asfreq('D').plot(ax=ax[0], marker='o')

data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
ax[1].legend(['back-fill', 'forward-fill']);

fig, ax = plt.subplots(3, figsize=(15, 10), sharey=True)
goog = goog.asfreq('D', method='pad')
goog.plot(ax=ax[0])
goog.shift(1000).plot(ax=ax[1])
goog.tshift(1000).plot(ax=ax[2])

local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')

ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=.3, color='red')

ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=.3, color='red')

ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=.3, color='red')

ROI=100*(goog.tshift(-365)/goog - 1)
ROI.plot()
plt.ylabel('% Return on Investment');

# rollling window
rolling = goog.rolling(365, center=True)
data = pd.DataFrame({'input' : goog, 
                     'one-year rolling_mean' : rolling.mean(),
                     'one-year rolling_std' : rolling.std()})

ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)


data = pd.read_csv('./data/FremontBridge.csv', index_col = 'Date', parse_dates=True)
data.head()    
data.columns= ['West', 'East']    
data.head()    
data.tail()
data['Total']= data.eval('West+East')    
data.dropna().describe()    
    
import seaborn; seaborn.set()    
data.plot(figsize = (15, 5))    
plt.ylabel('Hourly Bicyle Count')

weekly = data.resample('W').sum()
weekly.plot(style = [':', '--', '-'], figsize = (15, 4))
plt.ylabel('Weekly Bicyle Count')

daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'], figsize = (15, 4))
plt.ylabel('mean hourly count')

by_time = data.groupby(data.index.time).mean()
#by_time.head()
hourly_ticks= 4*60*60*np.arange(6)  # why 4*60*60*np.arange(6)??
hourly_ticks
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'], figsize=(15, 4))

by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.head()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
by_weekday.plot(style=[':', '--', '-'], figsize=(15, 4))

weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()
by_time.head()
by_time.tail()

fig, ax= plt.subplots(1, 2, figsize=(14, 5))
by_time.ix['Weekday'].plot(ax=ax[0], title='Weekdays', xticks=hourly_ticks, 
          style=[':', '--', '-'])
by_time.ix['Weekend'].plot(ax=ax[1], title='Weekendx', xticks=hourly_ticks, 
          style=[':', '--', '-']);
          
# eval(), query()
rng = np.random.RandomState(42)
x = rng.rand(10000000)
y = rng.rand(10000000)

mask = (x>0.5) & (y<0.5)
tmp1 = (x>0.5)
tmp2 = (y<0.5)
mask = tmp1 & tmp2

import numexpr
mask_numexpr = numexpr.evaluate('(x>0.5) & (y<0.5)')
np.allclose(mask, mask_numexpr)

nrows, ncols = 10000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))
df1.head()

df1 + df2 + df3 + df4
pd.eval('df1 + df2 + df3 + df4')
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0,1000, (100,3))) for i in range(5))
result1 = -df1*df2/(df3+df4) - df5
result2 = pd.eval('-df1*df2/(df3+df4) - df5')
np.allclose(result1, result2)
result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)
result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1, result3)
result1 = df2.T[0] + df2.iloc[1]
result2 = pd.eval('df2.T[0] + df2.iloc[1]')
np.allclose(result1, result2)

df = pd.DataFrame(rng.rand(1000, 3), columns = ['A', 'B', 'C'])
df.head()
result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval('(df.A + df.B) / (df.C - 1)')
np.allclose(result1, result2)
result3 = df.eval('(A+B) / (C-1)')
np.allclose(result1, result3)

df.head()
df.eval('D = (A + B)/C', inplace=True)
df.head()
df.eval('D=(A-B)/C', inplace=True)
df.head()

df.head()
df.mean(1)
column_mean = df.mean(1) # .mean(axis=1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean') # '@'only can be used in DataFrame.eval()
np.allclose(result1, result2)

result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1, result2)

result2 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result2)
Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')
np.allclose(result1, result2)
