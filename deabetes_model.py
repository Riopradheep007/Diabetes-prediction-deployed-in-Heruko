#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:06:16 2020

@author: kpr
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("processed_dataset.csv")
print(df.head())
sns.countplot(x="Outcome", data=df)

x=df.drop(['Outcome'],axis=1)

y=df['Outcome']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


model=RandomForestClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,prediction)

print("confusion_matrix:",accuracy)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,prediction)
print("accuracy_score:",accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))

import pickle

# Creating a pickle file for the classifier
filename = 'diabetes_prediction.pkl'
pickle.dump(model, open(filename, 'wb'))
'''
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc

fpr,tpr,threshold=roc_curve(y_test,prediction)
auc_curve=auc(fpr,tpr)

plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr,tpr,linestyle='-',label='Logistic(auc=%0.3f)'%auc_curve)

plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()
plt.show()
'''