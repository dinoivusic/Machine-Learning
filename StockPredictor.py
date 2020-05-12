#!/usr/bin/env python
# coding: utf-8

# In[50]:


#Stock prices prediction program using Machine Learning
import quandl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[51]:


#Stock data
df = quandl.get('WIKI/AMZN')
#df = quandl.get('WIKI/FB')
# print data
df.head()


# In[52]:


#Get adj closed
df = df[['Close']]
df.head()


# In[53]:


# A variable for prediction of n days out into future
forecast_out = 30 
#Create target column for dependent variable shifted n units up
df['Prediction'] = df[['Close']].shift(-forecast_out)
#print new dataset
df.tail()


# In[54]:


#Create independent dataset
#Convert DF to numpy array
X = np.array(df.drop(['Prediction'],1))
#Remove last n rown
X = X[:-forecast_out]
print(X)


# In[55]:


# Create dependent dataset y 
# convert df to a numpy arr
y = np.array(df['Prediction'])
# Get all y vlaues except last n rows
y = y[:-forecast_out]
y


# In[56]:


# split data into 80% train and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)


# In[57]:


# Create and train model SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)


# In[58]:


#Test model: Score return coeficient of determination R`2 of prediction
svm_conf = svr_rbf.score(X_test, y_test)
print('svm conf:', svm_conf)


# In[59]:


lr = LinearRegression()
#Train this model
lr.fit(X_train,y_train)


# In[60]:


#Test model: Score return coeficient of determination R`2 of prediction
lr_conf = lr.score(X_test, y_test)
print('lr conf:', lr_conf)
lr_conf


# In[61]:


#SEt x_forecast equal to last 30 rows of original dataset from Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
x_forecast


# In[62]:


#Print linear regression model predictions for n newxt days
lr_prediction = lr.predict(x_forecast)
lr_prediction

#Print SVM regressor for n days
svm_prediction = svr_rbf.predict(x_forecast)
svm_prediction


# In[ ]:




