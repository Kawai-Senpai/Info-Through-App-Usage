#!/usr/bin/env python
# coding: utf-8

# 
# # Gender Traciking with App Usage
# 
# Project by Ranit Bhowmick (https://linktr.ee/ranitbhowmick) and Sayanti Chatterjee (https://linktr.ee/sayantichatterjee)
# • Github link to this project : https://github.com/Kawai-Senpai/Info-Through-App-Usage
# 

# In[196]:


import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts


# In[197]:


# We have constructed the dataset custom through a survey.
# Here is the survey link : https://forms.gle/gQFGemdu8aciNnZp6 (Take the survey to help create a stronger dataset)

# Github : 

data = pd.read_csv("E:/python/AI_datasets/app_usage.csv")
dataf = data.drop(['Timestamp','Name ( Optional )'],axis='columns')
print(dataf) #Displaying the last few datas
print(dataf.dtypes) #Displaying the datatype of the datas


# In[198]:


#Outlier detection and correction function

def timeerror(x):
    x[x.str.slice(start=0,stop=2).astype(np.int64)>=24]='00:00:00'
    return x


# In[199]:


#Pre Processing App Usage Durtion Data

print('App Usage Durtion Data')

dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']]=dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']].fillna('00:00:00')
dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']] = dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']].apply(lambda x: timeerror(x))
dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']] = dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']].apply(lambda x: pd.to_datetime(x,format='%H:%M:%S'))

dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']]


# In[200]:


#App Usage Durtion Data to numerical value

dataf['zero'] = '00:00:00'
dataf['zero'] = pd.to_datetime(dataf['zero'],format='%H:%M:%S')

dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']] = dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']].apply(lambda x: (x-dataf['zero']).dt.total_seconds().astype(int))
print(dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage']])


# In[201]:


#Pre Processing App Time Durtion Data

print('App Usage Time Data')

dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']]=dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']].fillna('00:00')
dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']] = dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']].apply(lambda x: pd.to_datetime(x,format='%H:%M'))

dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']]


# In[202]:


#App Usage Durtion Data to numerical value

dataf['zero'] = '00:00'
dataf['zero'] = pd.to_datetime(dataf['zero'],format='%H:%M')

dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']] = dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']].apply(lambda x: (x-dataf['zero']).dt.total_seconds().astype(int))
print(dataf[['Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']])


# In[203]:


#Convert Date of Birth

dataf['Date of Birth'] = (pd.to_datetime("today")-pd.to_datetime(dataf['Date of Birth'],format='%Y-%m-%d')).dt.days
print(dataf['Date of Birth'])


# In[204]:


#One Hot Encoding

dummies = pd.get_dummies(dataf[['Gender','Employment Status','Field']])
dataf = pd.concat([dataf,dummies],axis='columns')
dataf = dataf.drop(['Gender','Employment Status','Field'],axis='columns')
print(dataf)


# In[267]:


#Train Test Split - Gender Prediction

x_train,x_test,y_train,y_test = tts(dataf[['Transportation Usage','Social Usage','Meet Usage','Shopping Usage','Food Usage','Trading Usage','Gaming Usage','Banking Usage','Networking Usage','Cinema Usage','Transportation Time','Social Time','Meet Time','Shopping Time','Food Time','Trading Time','Gaming Time','Banking Time','Networking Time','Cinema Time']],dataf[['Gender_Female','Gender_Male']],test_size=0.2)
print("X Train Size = ",x_train.shape)
print("Y Train Size = ",y_train.shape)
print("X Test Size = ",x_test.shape)
print("Y Test Size = ",y_test.shape)


# In[236]:


#Using Discision Trees

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

model.score(x_test,y_test)


# In[282]:


#Using Random Forest

from sklearn.ensemble import RandomForestClassifier as rf

model2 = rf(n_estimators=15)
model2.fit(x_train,y_train)

model2.score(x_test,y_test)


# In[ ]:


#With Random Forest we are able to reach upto 82% accuracy

