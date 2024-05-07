#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd 
df = pd.read_csv("heartDisease.csv")
df.head()


# In[48]:


len(df)


# In[90]:


df['HeartDisease'].value_counts()


# In[49]:


df.isna().sum() 


# In[73]:


df = df.sample(frac = 0.3)


# In[74]:


len(df)


# In[77]:


df[df.columns[1]].dtype


# In[78]:


for c in range(1, len(df.columns)):
    if df[df.columns[c]].dtype == 'object':
        #print(df[df.columns[c]])
        df[df.columns[c]] = pd.Categorical(df[df.columns[c]])
        df[df.columns[c]] = df[df.columns[c]].cat.codes 


# In[105]:


df.head()


# In[106]:


X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']


# In[107]:


X


# In[108]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)


# In[109]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)


# In[110]:


clf.score(X_test, y_test)


# In[111]:


clf.score(X_train, y_train)


# In[95]:


from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[113]:


from sklearn.model_selection import GridSearchCV 
gbm = GradientBoostingClassifier()
parameters = {'learning_rate': [0.2, 0.1, 0.05],
             'max_depth': [3, 4, 5]
             }
clf = GridSearchCV(gbm, parameters)
clf.fit(X_train, y_train)


# In[114]:


clf.score(X_test, y_test)


# In[115]:


clf.score(X_train, y_train)


# In[116]:


clf.best_params_


# In[117]:


y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[96]:


gbm = GradientBoostingClassifier()
parameters = {'learning_rate': [0.2, 0.1, 0.05],
              'n_estimators': [90, 100],
             'max_depth': [3, 4, 5],
              'min_samples_leaf': [1, 3]
             }
clf = GridSearchCV(gbm, parameters)
clf.fit(X_train, y_train)


# In[97]:


clf.score(X_test, y_test)


# In[98]:


clf.score(X_train, y_train)


# In[99]:


clf.best_params_


# In[101]:


y_pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp


# In[ ]:




