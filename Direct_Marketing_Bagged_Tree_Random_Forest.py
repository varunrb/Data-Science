#!/usr/bin/env python
# coding: utf-8

# ## Direct Marketing Bagged Tree Classifier

# In[64]:


import pandas as pd
from pandas import DataFrame
import numpy as np

import os
os.getcwd()


# In[65]:


dm = pd.read_csv('directmarketing.csv',sep=',',header=0)
dm.head()


# In[66]:


dm.info()


# In[67]:


# #--According to the marketer customer who spends more than the average spend is considered as Good--#
# Target variable
dm['Target'] = (dm.AmountSpent>dm.AmountSpent.mean()).map({True:1,False:0})
dm.head()


# In[68]:


# Missing values
dm.isnull().sum()


# In[69]:


# Create separate category 'Missing' for History 
dm['History'] = dm.History.replace({np.nan:'Missing'})
print(dm.History.unique())
print(dm.isnull().sum())
dm.head()


# In[70]:


# Converting Children and Catalogs into categorical variables
dm['Children'] = dm.Children.astype('category')
dm['Catalogs'] = dm.Catalogs.astype('category')
dm.dtypes


# In[71]:


# Target variable and Predictor variables
y = dm['Target']
y.head()


# In[72]:


X = dm.drop(['History','AmountSpent','Cust_Id','Target'],axis=1)
X.head()


# In[73]:


# Dummy variables
X = pd.get_dummies(X)
X.head()


# In[74]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=200)


# In[75]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[76]:


clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=20,random_state=200,
                      base_estimator=DecisionTreeClassifier())


# In[77]:


clf.fit(X_train,y_train)


# In[78]:


clf.oob_score_


# In[79]:


clf.score(X_test,y_test)


# In[80]:


# Alternative to K-fold CV
for w in range(10,300,20):
    clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=w,random_state=200,
                          base_estimator=DecisionTreeClassifier())
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')


# In[81]:


#Finalizing on a tree model with 210 trees
clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=210,random_state=200,
                      base_estimator=DecisionTreeClassifier())
clf.fit(X_train,y_train)


# In[82]:


clf.oob_score_


# In[83]:


clf.score(X_test,y_test)


# In[84]:


# AUC
import sklearn.metrics as metrics
metrics.roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])


# In[86]:


y_pred=clf.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)


# In[87]:


print(clf.estimators_[0].feature_importances_) #For 1st tree [0]


# In[88]:


# We can extract feature importance from each tree then take a mean for all trees
imp=[]
for i in clf.estimators_:
    imp.append(i.feature_importances_)
imp=np.mean(imp,axis=0)


# In[89]:


feature_importance=pd.Series(imp,index=X.columns.tolist())


# In[90]:


feature_importance.sort_values(ascending=False)


# In[91]:


feature_importance.sort_values(ascending=False).plot(kind='bar')


# ### Random Forest Classifier

# In[92]:


from sklearn.ensemble import RandomForestClassifier


# In[93]:


clf2 =RandomForestClassifier(n_estimators=20,oob_score=True,n_jobs=-1,random_state=200)


# In[94]:


clf2.fit(X_train,y_train)


# In[95]:


clf2.oob_score_


# In[96]:


clf2.score(X_test,y_test)


# In[97]:


for w in range(10,300,20):
    clf2=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=200)
    clf2.fit(X_train,y_train)
    oob=clf2.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')


# In[98]:


#Finalize 70 trees
clf2=RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=200)


# In[99]:


clf2.fit(X_train,y_train)


# In[100]:


clf2.oob_score_


# In[101]:


clf2.score(X_test,y_test)


# In[102]:


import sklearn.metrics as metrics
metrics.roc_auc_score(y_test,clf2.predict_proba(X_test)[:,1])


# In[103]:


y_pred2=clf2.predict(X_test)
metrics.confusion_matrix(y_test,y_pred2)


# In[104]:


clf2.feature_importances_


# In[105]:


imp_feat=pd.Series(clf2.feature_importances_,index=X.columns.tolist())


# In[106]:


imp_feat.sort_values(ascending=False)


# In[107]:


imp_feat.sort_values(ascending=False).plot(kind='bar')

