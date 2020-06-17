
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

import pickle5

# Creating a pickle file for the classifier
filename = 'diabetes_prediction.pkl'
pickle5.dump(model, open(filename, 'wb'))
