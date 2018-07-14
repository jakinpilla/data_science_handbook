# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 23:22:15 2018

@author: Daniel
"""

# pandas
from os import getcwd, chdir
getcwd()
#chdir('C:/Users/dsc/data_science_handbook')
chdir('C:/Users/daniel/data_science_handbook')

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
data['a':'c']
data[0:2]
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
pd.concat([df1, df2], axis=1)
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










