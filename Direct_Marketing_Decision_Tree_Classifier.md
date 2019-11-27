
# Direct Marketing Decision Tree Classifier

To build a decision tree model to classify customers as Good/Bad based on amount spent.
Good - if amount spent is >average amount spent


```python
import pandas as pd
from pandas import DataFrame
import numpy as np

import os
os.getcwd()
```




    'C:\\Users\\Varun R Bhat\\Downloads\\Jigsaw\\Direct_Marketing_Data_Set'




```python
dm = pd.read_csv('directmarketing.csv',sep=',',header=0)
dm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>OwnHome</th>
      <th>Married</th>
      <th>Location</th>
      <th>Salary</th>
      <th>Children</th>
      <th>History</th>
      <th>Catalogs</th>
      <th>AmountSpent</th>
      <th>Cust_Id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Old</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Far</td>
      <td>47500</td>
      <td>0</td>
      <td>High</td>
      <td>6</td>
      <td>755</td>
      <td>247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>63600</td>
      <td>0</td>
      <td>High</td>
      <td>6</td>
      <td>1318</td>
      <td>127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Young</td>
      <td>Female</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>13500</td>
      <td>0</td>
      <td>Low</td>
      <td>18</td>
      <td>296</td>
      <td>479</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Own</td>
      <td>Married</td>
      <td>Close</td>
      <td>85600</td>
      <td>1</td>
      <td>High</td>
      <td>18</td>
      <td>2436</td>
      <td>475</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Middle</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Close</td>
      <td>68400</td>
      <td>0</td>
      <td>High</td>
      <td>12</td>
      <td>1304</td>
      <td>151</td>
    </tr>
  </tbody>
</table>
</div>




```python
dm.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 11 columns):
    Age            1000 non-null object
    Gender         1000 non-null object
    OwnHome        1000 non-null object
    Married        1000 non-null object
    Location       1000 non-null object
    Salary         1000 non-null int64
    Children       1000 non-null int64
    History        697 non-null object
    Catalogs       1000 non-null int64
    AmountSpent    1000 non-null int64
    Cust_Id        1000 non-null int64
    dtypes: int64(5), object(6)
    memory usage: 86.0+ KB
    


```python
# #--According to the marketer customer who spends more than the average spend is considered as Good--#
# Target variable
dm['Target'] = (dm.AmountSpent>dm.AmountSpent.mean()).map({True:1,False:0})
dm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>OwnHome</th>
      <th>Married</th>
      <th>Location</th>
      <th>Salary</th>
      <th>Children</th>
      <th>History</th>
      <th>Catalogs</th>
      <th>AmountSpent</th>
      <th>Cust_Id</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Old</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Far</td>
      <td>47500</td>
      <td>0</td>
      <td>High</td>
      <td>6</td>
      <td>755</td>
      <td>247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>63600</td>
      <td>0</td>
      <td>High</td>
      <td>6</td>
      <td>1318</td>
      <td>127</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Young</td>
      <td>Female</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>13500</td>
      <td>0</td>
      <td>Low</td>
      <td>18</td>
      <td>296</td>
      <td>479</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Own</td>
      <td>Married</td>
      <td>Close</td>
      <td>85600</td>
      <td>1</td>
      <td>High</td>
      <td>18</td>
      <td>2436</td>
      <td>475</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Middle</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Close</td>
      <td>68400</td>
      <td>0</td>
      <td>High</td>
      <td>12</td>
      <td>1304</td>
      <td>151</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Missing values
dm.isnull().sum()
```




    Age              0
    Gender           0
    OwnHome          0
    Married          0
    Location         0
    Salary           0
    Children         0
    History        303
    Catalogs         0
    AmountSpent      0
    Cust_Id          0
    Target           0
    dtype: int64




```python
# Create separate category 'Missing' for History 
dm['History1'] = dm.History.replace({np.nan:'Missing'})
print(dm.History1.unique())
print(dm.isnull().sum())
dm.head()
```

    ['High' 'Low' 'Medium' 'Missing']
    Age              0
    Gender           0
    OwnHome          0
    Married          0
    Location         0
    Salary           0
    Children         0
    History        303
    Catalogs         0
    AmountSpent      0
    Cust_Id          0
    Target           0
    History1         0
    dtype: int64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>OwnHome</th>
      <th>Married</th>
      <th>Location</th>
      <th>Salary</th>
      <th>Children</th>
      <th>History</th>
      <th>Catalogs</th>
      <th>AmountSpent</th>
      <th>Cust_Id</th>
      <th>Target</th>
      <th>History1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Old</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Far</td>
      <td>47500</td>
      <td>0</td>
      <td>High</td>
      <td>6</td>
      <td>755</td>
      <td>247</td>
      <td>0</td>
      <td>High</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>63600</td>
      <td>0</td>
      <td>High</td>
      <td>6</td>
      <td>1318</td>
      <td>127</td>
      <td>1</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Young</td>
      <td>Female</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>13500</td>
      <td>0</td>
      <td>Low</td>
      <td>18</td>
      <td>296</td>
      <td>479</td>
      <td>0</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Own</td>
      <td>Married</td>
      <td>Close</td>
      <td>85600</td>
      <td>1</td>
      <td>High</td>
      <td>18</td>
      <td>2436</td>
      <td>475</td>
      <td>1</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Middle</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Close</td>
      <td>68400</td>
      <td>0</td>
      <td>High</td>
      <td>12</td>
      <td>1304</td>
      <td>151</td>
      <td>1</td>
      <td>High</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Converting Children and Catalogs into categorical variables
dm['Children'] = dm.Children.astype('category')
dm['Catalogs'] = dm.Catalogs.astype('category')
dm.dtypes
```




    Age              object
    Gender           object
    OwnHome          object
    Married          object
    Location         object
    Salary            int64
    Children       category
    History          object
    Catalogs       category
    AmountSpent       int64
    Cust_Id           int64
    Target            int64
    History1         object
    dtype: object




```python
# Target variable and Predictor variables
y = dm['Target']
y.head()
```




    0    0
    1    1
    2    0
    3    1
    4    1
    Name: Target, dtype: int64




```python
X = dm.drop(['History','AmountSpent','Cust_Id','Target'],axis=1)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>OwnHome</th>
      <th>Married</th>
      <th>Location</th>
      <th>Salary</th>
      <th>Children</th>
      <th>Catalogs</th>
      <th>History1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Old</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Far</td>
      <td>47500</td>
      <td>0</td>
      <td>6</td>
      <td>High</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>63600</td>
      <td>0</td>
      <td>6</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Young</td>
      <td>Female</td>
      <td>Rent</td>
      <td>Single</td>
      <td>Close</td>
      <td>13500</td>
      <td>0</td>
      <td>18</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Middle</td>
      <td>Male</td>
      <td>Own</td>
      <td>Married</td>
      <td>Close</td>
      <td>85600</td>
      <td>1</td>
      <td>18</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Middle</td>
      <td>Female</td>
      <td>Own</td>
      <td>Single</td>
      <td>Close</td>
      <td>68400</td>
      <td>0</td>
      <td>12</td>
      <td>High</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dummy variables
X = pd.get_dummies(X)
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Salary</th>
      <th>Age_Middle</th>
      <th>Age_Old</th>
      <th>Age_Young</th>
      <th>Gender_Female</th>
      <th>Gender_Male</th>
      <th>OwnHome_Own</th>
      <th>OwnHome_Rent</th>
      <th>Married_Married</th>
      <th>Married_Single</th>
      <th>...</th>
      <th>Children_2</th>
      <th>Children_3</th>
      <th>Catalogs_6</th>
      <th>Catalogs_12</th>
      <th>Catalogs_18</th>
      <th>Catalogs_24</th>
      <th>History1_High</th>
      <th>History1_Low</th>
      <th>History1_Medium</th>
      <th>History1_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47500</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63600</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85600</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68400</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
# Decision Tree Model
import sklearn.model_selection as model_selection
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=200)
```


```python
import sklearn.tree as tree
clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
clf.fit(X_train,y_train)
clf.score(X_test,y_test) # Accuracy Score
```




    0.8533333333333334




```python
# AUC value
import sklearn.metrics as metrics
metrics.roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
```




    0.895586062252729




```python
import pydotplus
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=X.columns,  
                         class_names=["0","1"],  
                         filled=True, rounded=True,  
                         special_characters=True,proportion=True)
```


```python
graph = pydotplus.graph_from_dot_data(dot_data)
```


```python
from IPython.display import Image
```


```python
Image(graph.create_png())
```




![png](output_18_0.png)



### Grid Search-Cross Validation


```python
clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
```


```python
mod=model_selection.GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5,6]},cv=5)
mod.fit(X_train,y_train)
```




    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=DecisionTreeClassifier(class_weight=None,
                                                  criterion='gini', max_depth=3,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort=False, random_state=200,
                                                  splitter='best'),
                 iid='warn', n_jobs=None, param_grid={'max_depth': [2, 3, 4, 5, 6]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
mod.best_estimator_
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=200, splitter='best')




```python
# Running model with max_depth=5 as per best_estimator_
clf2=tree.DecisionTreeClassifier(max_depth=5,random_state=200)
clf2.fit(X_train,y_train)
clf2.score(X_test,y_test) 
```




    0.8566666666666667




```python
# AUC value
import sklearn.metrics as metrics
metrics.roc_auc_score(y_test,clf2.predict_proba(X_test)[:,1])
```




    0.9055484055484055




```python
y_pred = clf2.predict(X_test)
```


```python
metrics.confusion_matrix(y_test,y_pred)
```




    array([[169,  20],
           [ 23,  88]], dtype=int64)


