#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv("Diabetes.csv")
dataset.head(11)


# In[3]:


dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:


X = dataset.iloc[:, : -1]
y = dataset.iloc[:, -1]


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 25, random_state = 0)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score
acc_log_1 = round(accuracy_score(y_pred, y_test), 2) * 100
print('Accuracy: {0}'.format(acc_log_1))


# In[23]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log_2 = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy: {score}".format(score = acc_log_2))


# In[26]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_log_3 = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy: {score}".format(score = acc_log_3))


# In[ ]:




