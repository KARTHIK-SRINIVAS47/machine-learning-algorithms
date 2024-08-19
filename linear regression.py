#!/usr/bin/env python
# coding: utf-8

# # implementation of  linear regression in callifornia_housing dataset

# In[3]:


from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=housing()


# In[7]:


type(df)


# In[11]:


df


# In[12]:


#creating dataframe wiht columns and seeing first 5 rows
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset.head()


# In[13]:


#bringing target price into the frame 
dataset['price']=df.target
dataset.head()


# In[15]:


#dividing the dataset into independent and dependent features
x=dataset.iloc[:,:-1]##independent features
y=dataset.iloc[:,-1]##dependent features


# In[17]:


x.head()



# In[18]:


y.head()


# In[27]:


##linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg=LinearRegression()
mse=cross_val_score(lin_reg,x,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[ ]:





# In[ ]:




