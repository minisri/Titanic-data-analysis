#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


df=pd.read_csv(r'C:\Users\ananyya srivastava\Downloads\train (1).csv',index_col=0)


# In[48]:


df=df.dropna()


# In[49]:


df.head()


# In[50]:


df.info()


# In[51]:


df.shape


# In[52]:


df.columns


# In[53]:


df.describe()


# In[54]:


df.Survived.value_counts()


# In[55]:


sns.boxplot(x=df.Age)


# In[56]:


df.Embarked.value_counts()


# In[57]:


df.Pclass.value_counts()


# In[58]:


sns.countplot(x='Pclass', data=df, label='Count')


# In[59]:


sns.countplot(x='Survived',data=df,label='count')


# In[60]:


df.drop(['Ticket','Cabin'],axis=1,inplace=True)


# In[61]:


df.head()


# In[62]:


sns.boxplot(x=df.Fare)


# In[63]:


sns.countplot(x=df.Sex)


# In[64]:


df.Sex.value_counts()


# In[65]:


t1=pd.crosstab(df.Sex,df.Survived)


# In[66]:


t1


# In[67]:


t1.plot(kind='bar')


# In[68]:


t2=pd.crosstab(df.Embarked,df.Survived)


# In[69]:


t2


# In[70]:


t2.plot(kind='bar')


# In[71]:


t3=pd.crosstab(df.Pclass,df.Survived)


# In[72]:


t3


# In[73]:


t3.plot(kind='bar')


# In[74]:


df.drop(['Name'],axis=1,inplace=True)


# In[75]:


df.head()


# In[76]:


map1={'female':0,'male':1}
df['gender']=df['Sex'].map(map1)


# In[77]:


df.head()


# In[79]:


df.drop(['Sex'],axis=1,inplace=True)


# In[80]:


df.head()


# In[83]:


c=df.corr()


# In[84]:


sns.heatmap(c)


# In[85]:


sns.distplot(df.Age,bins=20)


# In[87]:


sns.distplot(df.Fare,bins=50)


# In[97]:


map2={'S':0,'Q':1,'C':2}
df['em']=df['Embarked'].map(map2)


# In[98]:


df.head()


# In[101]:


df.drop(['Embarked'],axis=1,inplace=True)


# In[102]:


df.head()


# In[125]:


dft=pd.read_csv(r'C:\Users\ananyya srivastava\Downloads\test (1).csv',index_col=0)


# In[ ]:





# In[126]:


dft.head()


# In[127]:


dft=dft.dropna()


# In[128]:


dft.info()


# In[129]:


dft.columns


# In[130]:


dft.shape


# In[131]:


dft.describe()


# In[132]:


map3={'S':0,'Q':1,'C':2}
dft['em']=dft['Embarked'].map(map3)


# In[133]:


dft.drop(['Embarked'],axis=1,inplace=True)


# In[134]:


dft.head()


# In[ ]:





# In[135]:


map4={'female':0,'male':1}
dft['gender']=dft['Sex'].map(map4)


# In[136]:


dft.drop(['Sex'],axis=1,inplace=True)


# In[137]:


dft.head()


# In[140]:


dft.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)


# In[141]:


dft.head()


# In[144]:


X_train = df.drop("Survived", axis=1)
Y_train = df["Survived"]

X_train.shape, Y_train.shape


# In[152]:


test_x= dft[X_train.columns.values] #makes it same size as train_x column wise


# In[154]:


test_x.head()


# In[156]:


test_x.shape


# In[165]:


gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(test_x)

gaussian.score(X_train, Y_train)


# In[160]:


from sklearn.naive_bayes import GaussianNB


# In[168]:



from sklearn.ensemble import RandomForestClassifier 
random_forest = RandomForestClassifier(n_estimators=100)
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features='sqrt', min_samples_split=5)

random_forest.fit(X_train, Y_train)

Y_pred_1 = random_forest.predict(test_x)

random_forest.score(X_train, Y_train)


# In[ ]:




