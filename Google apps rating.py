#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


google_data = pd.read_csv('googleplaystore.csv')


# In[9]:


google_data.head(10)    


# In[5]:


google_data.shape


# In[6]:


google_data.describe()


# In[7]:


google_data.boxplot() #here is an outlier above 17.5


# In[8]:


google_data.hist()  #look its left skeewed hencde we use median


# In[9]:


google_data.info() #total row sholud be 10841 but sme has less ,hnce they r null values


# In[10]:


google_data.isnull()


# In[11]:


# Count the number of missing values in each column
google_data.isnull().sum()


# In[12]:


#Check how many ratings are more than 5 - Outliers
google_data[google_data.Rating > 5]


# In[13]:


#removing the above row ie the outlier value

google_data.drop([10472],inplace=True)


# In[14]:


google_data[10470:10475]#just checking if that row is removed


# In[15]:


google_data.boxplot()#here data is concentrated where the box is present i,e 4-5


# In[16]:


google_data.hist() #here its right sweed hence use median to fill data


# In[17]:


#Remove columns that are 90% empt
threshold = len(google_data)* 0.1
threshold


# In[18]:



google_data.dropna(thresh=threshold, axis=1, inplace=True)


# In[19]:


print(google_data.isnull().sum())


# In[20]:


#Data Imputation and Manipulation
#Fill the null values with appropriate values using aggregate functions such as mean, median or mode.
#Define a function impute_median
def impute_median(series):
    return series.fillna(series.median())


# In[21]:


google_data.Rating = google_data['Rating'].transform(impute_median)


# In[22]:



#count the number of null values in each column
google_data.isnull().sum()


# In[23]:



# modes of categorical values
print(google_data['Type'].mode())
print(google_data['Current Ver'].mode())
print(google_data['Android Ver'].mode())
#here we get unimodal vale i.e single value like free henvce easy to use but if it would had 2 values or bimodal valuesthen
#we would had taken the first value from the 2 value to be the mode


# In[24]:


#
# Fill the missing categorical values with mode
google_data['Type'].fillna(str(google_data['Type'].mode().values[0]), inplace=True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]), inplace=True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]), inplace=True)


# In[25]:


#count the number of null values in each column
google_data.isnull().sum()
#hence no null value


# In[26]:


### Let's convert Price, Reviews and Ratings into Numerical Values
google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
google_data['Price'] = google_data['Price'].apply(lambda x: float(x))
google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors='coerce')
#coerce =ignore 


# In[27]:


google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: float(x))


# In[28]:



google_data.head(10)


# In[29]:


google_data.describe()#previously we got only one column now we got 4 on whinch we can PPLY THE STATS


# In[30]:



grp = google_data.groupby('Category')#all games ,movie...etc apps together
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)


# In[31]:


plt.figure(figsize=(12,5))
plt.plot(x, "ro", color='g')
plt.xticks(rotation=90)
plt.show()


# In[32]:


plt.figure(figsize=(16,5))
plt.plot(x,'ro', color='r')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Rating-->')
plt.show()


# In[33]:


plt.figure(figsize=(16,5))
plt.plot(y,'r--', color='b')
plt.xticks(rotation=90)
plt.title('Category wise Pricing')
plt.xlabel('Categories-->')
plt.ylabel('Prices-->')
plt.show()


# In[34]:


plt.figure(figsize=(16,5))
plt.plot(z,'bs', color='g')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()


# In[ ]:




