
# coding: utf-8

# ## Hyper Parameter Tuning

# - In contrast to __model parameters__ which are learned during training, __model hyperparameters__ are set by the data scientist ahead of training and control implementation aspects of the model. 
# - The __weights learned during training__ of a linear regression model are __parameters__ while the __number of trees in a random forest is a model hyperparameter__ because this is set by the data scientist. 
# - __Hyperparameters__ can be thought of as __model settings__. These settings need to be tuned for each problem because the best model hyperparameters for one particular dataset will not be the best across all datasets. 
# - The process of hyperparameter tuning (also called __hyperparameter optimization)__ means finding the combination of hyperparameter values for a machine learning model that performs the best - as measured on a validation dataset - for a problem.

# In[ ]:


##! pip freeze


# In[ ]:


##! pip install -U scikit-learn


# ### Logistic Regression Model

# In[1]:


# Data manipulation libraries
import pandas as pd
import numpy as np

##### Scikit Learn modules needed for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Plotting libraries
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("titanic/train.csv")
df.head()


# In[ ]:


print(df.describe())
df.isna().sum()


# In[3]:


# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['Embarked', 'Sex', 'Pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])


# In[ ]:


df.columns


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(df[['Fare','Pclass', 'Name', 'Sex', 'Age','Embarked']], 
                                                    df["Survived"], test_size=0.2)


# In[5]:


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))


# Reference on Grid Search
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

# In[6]:


LogisticRegression()


# In[ ]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__solver': ["newton-cg","lbfgs", "liblinear", "sag", "saga"]
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best Logistic Regression from grid search: %.3f"
       % grid_search.score(X_test, y_test)))


# In[10]:


grid_search.best_params_

