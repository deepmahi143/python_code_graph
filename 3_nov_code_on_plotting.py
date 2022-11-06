#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[2]:


9+9


# In[3]:


import seaborn as sns


# In[5]:


import seaborn as sns
df = sns.load_dataset('tips')
df.head()


# In[7]:


# count plot code
sns.countplot(x = 'day', data = df)


# In[8]:


sns.countplot(x = 'sex', data = df)


# In[9]:


sns.countplot(x = 'time', data = df)


# In[10]:


sns.countplot(x = 'size', data = df)


# In[13]:


# box plot
# importing numpy library

import numpy as np
np.random.seed(7)
data = np.random.normal(100,20,200)
plt.figure(figsize = (8,6))
plt.boxplot(data)
plt.show()


# In[14]:


pip install matplotlib


# In[15]:


import matplotlib.pyplot as plt


# In[17]:


import numpy as np

#creating data
#creating random size
np.random.seed(7)
data = np.random.normal(100,20,200)

#setting plot size
plt.figure(figsize = (8,6))

#creating boxplot
plt.boxplot(data)
plt.show()


# In[22]:


# histogram
# creating the data
data = [1.7,1.8,7.8,2.3,5.6,4.5,6.7,8.9,9.8,3.3,5.5,2.3,5.6,2.3]
# this histogram has avlues from 1 to 9
plt.hist(data,bins = 2)
plt.show()


# In[25]:


# stacked bar chart
# creating the daaaaaata
x = ['A','B','C','D']
y1 = np.array([10,20,30,10])
y2 = np.array([20,27,25,37])
y3 = np.array([34,56,34,23])
y4 = np.array([23,45,66,67])

# plot
plt.bar(x,y1,color = 'r')
plt.bar(x,y2,bottom = y1,color = 'b')
plt.bar(x,y3, bottom = y1+y2,color = 'y')
plt.bar(x,y4,bottom = y1+y2+y3,color ='g')
plt.xlabel('Teams')
plt.ylabel('Score')
plt.legend(['Round 1','Round 2', 'Round 3', 'Round 4'])
plt.title('Scores by team in 4 rounds')
plt.show()



# In[26]:


# pie chart
labels =['python','c++','Ruby','JAVA']
sizes = [215,130,245,210]

#plot
plt.pie(sizes, labels = labels, autopct = '%1.2f%%', shadow = True, startangle =140)
plt.axis('equal')
plt.show()


# In[28]:


# heat map
# Creating data 
data = np.random.randint(low =1,
                         high =100, 
                         size = (10,10))
print('raw data:\n',data)


# In[29]:


# plotting heatmap


sns.heatmap(data = data, annot = True )


# In[30]:


sns.heatmap(data = data, annot = False )


# In[33]:


# cat plot
exercise = sns.load_dataset('exercise')
sns.catplot(x = 'time',y = 'pulse', hue = 'kind', data = exercise)


# In[34]:


# hue : encodes the points with different colour with respect to target variable


# In[37]:


# dist plot
df = sns.load_dataset('tips')
df.head()


# In[47]:


#distplot
sns.distplot(df['total_bill'], kde = True, color = 'green', bins = 10)


# In[46]:


sns.distplot(df['total_bill'], kde = False, color = 'green', bins = 10)


# In[45]:


# pair plot
sns.pairplot(df, hue = 'sex', palette = 'coolwarm')


# In[48]:


# pair plot
sns.pairplot(df, hue = 'smoker', palette = 'coolwarm')


# In[51]:


# join plot
sns.jointplot( x = 'total_bill', y = 'tip', data = df)


# In[52]:


# join plot
sns.jointplot( x = 'total_bill', y = 'tip', data = df, kind = 'kde')


# In[53]:


# join plot
sns.jointplot( x = 'total_bill', y = 'tip', data = df , kind ='hex')


# In[ ]:




