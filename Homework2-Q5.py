#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import cvxpy as cp
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics


# In[68]:


df=pd.read_csv("ionosphere.data",delimiter=",",header=None)
array=df.values
X=array[:,0:34]
Y=array[:,34]


# In[69]:


X_Train,X_Test,Y_Train,Y_Test=X[:300,:],X[300:,:],Y[:300],Y[300:]


# In[70]:


df_np=df.to_numpy()
Y_Modified=np.where(df_np[0:300,34]=='g', 1, -1)


# LEAST SQUARES TRAINING

# In[71]:


w1=cp.Variable(34)
b1=cp.Variable()
obj=0
for i in range(301):
    obj=cp.sum(((cp.multiply(Y_Modified, X_Train@w1+b1))-1)**2)
cp.Problem(cp.Minimize(obj), []).solve()
w1=w1.value
b1=b1.value


# LEAST SQUARES TESTING

# In[72]:


Y_TestModified=np.where(df_np[300:,34]=='g', 1, -1)
Y_Prediction=np.dot(X_Test,w1)+b1
Y_PredModified=np.where(Y_Prediction>0, 1, -1)

print('Accuracy for least squares:', metrics.accuracy_score(Y_TestModified,Y_PredModified,normalize=1))


# LOGISTIC LOSS TRAINING

# In[73]:


b2=cp.Variable()
w2=cp.Variable(34)


# LOGISTIC LOSS TESTING

# In[74]:


costFunction=cp.sum(cp.logistic(-cp.multiply(Y_Modified, X_Train@w2+b2)))
problem=cp.Problem(cp.Minimize(costFunction))
problem.solve(verbose=True, solver=cp.ECOS) 
prediction=X_Test@w2+b2
Y_Pred=prediction.value
Y_PredModified=np.where(Y_Pred[:]>0,1,-1)
print('Accuracy for logistic model: {}'.format(metrics.accuracy_score(Y_TestModified,Y_PredModified)))


# HINGE LOSS TRAINING

# In[75]:


b3=cp.Variable()
w3=cp.Variable(34)


# HINGE LOSS TESTING

# In[76]:


cost=cp.sum(cp.maximum(0,1- cp.multiply(Y_Modified, X_Train@w3+b3)))
prob=cp.Problem(cp.Minimize(cost))
prob.solve(verbose=True, solver=cp.ECOS) 
prediction=X_Test@w3+b3
Y_Pred=prediction.value
Y_PredModified=np.where(Y_Pred[:]>0,1,-1)
print('Accuracy for hinge loss model: {}'.format(metrics.accuracy_score(Y_TestModified,Y_PredModified)))


# In[ ]:




