#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:34:11 2020

@author: kpr
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
df=pd.read_csv("dataset.csv")
print(df.head())
print(df.isnull().sum())

x=df.drop(['Outcome'],axis=1)
y=df['Outcome']

#############Balance the dataset################################
from imblearn.combine import SMOTETomek
smk=SMOTETomek(random_state=42)
X_res,Y_res=smk.fit_sample(x,y)

m=pd.DataFrame()
m= pd.concat([X_res,Y_res],axis=1)
print(Y_res.shape,X_res.shape)

x=m.drop(['Outcome'],axis=1)
y=m['Outcome']

cs=m
cs.to_csv('../balanced_dataset.csv', index=False)
print('modified train CSV is ready!')

sns.countplot(x='Outcome',data=m)
##############################################################'''
#Outliers handling

df=pd.read_csv('balanced_dataset.csv')
print(df.head())
print(df.shape)
#sns.countplot(x='Outcome', data=df)

x=df.drop(['Outcome'],axis=1)
y=df['Outcome']

plt.hist(df.Pregnancies, bins=20, rwidth=0.8)
plt.xlabel('pregnancies(no of times)')
plt.ylabel('Count')
plt.show()

print(df.Pregnancies.min())
print(df.Pregnancies.max())
print(df.describe())



upper_limit = df.Pregnancies.mean() + 2*df.Pregnancies.std()
print(upper_limit)

lower_limit = df.Pregnancies.mean() -2*df.Pregnancies.std()
print(lower_limit)



print(df[(df.Pregnancies>upper_limit) | (df.Pregnancies<lower_limit)])
#remove outliers
removed_outliers=pd.DataFrame()
removed_outliers=df[(df.Pregnancies<upper_limit) & (df.Pregnancies>lower_limit)]
print(removed_outliers.shape)


'''
from scipy.stats import norm
import numpy as np
plt.hist(df.Pregnancies, bins=20, rwidth=0.8, density=True)
plt.xlabel('pregnancies(no of times)')
plt.ylabel('Count')

rng = np.arange(df.Pregnancies.min(), df.Pregnancies.max(), 0.1)
plt.plot(rng, norm.pdf(rng,df.Pregnancies.mean(),df.Pregnancies.std()))'''

sns.countplot(x='Age',data=removed_outliers)

print(removed_outliers.Age.max())
print(removed_outliers.Age.min())


upper_limit = removed_outliers.Age.mean() + 2*removed_outliers.Age.std()
print(upper_limit)

lower_limit = removed_outliers.Age.mean() -2*removed_outliers.Age.std()
print(lower_limit)



print(removed_outliers[(removed_outliers.Age>upper_limit) | (removed_outliers.Age<lower_limit)])


removed_outliers=removed_outliers[(removed_outliers.Age<upper_limit) & (removed_outliers.Age>lower_limit)]
print(removed_outliers.shape)
'''
cs=removed_outliers
cs.to_csv('../formulated_dataset1.csv', index=False)
print('modified train CSV is ready!')'''
