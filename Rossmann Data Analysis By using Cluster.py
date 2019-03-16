#!/usr/bin/env python
# coding: utf-8

# ####  31 Months of Data 
# #### Task 1: Subset the data for the first three months 
# #### Task 2: K Elbow ( on first 3 months data )
# #### Task 3: K Means ( Create Clusters )
# #### Task 4: Visualize the clusters ( Create Clusters )(Sample of observations)
# #### Task 5: The clusters formed in the step 3 needs to be tracked across next 30 Quarters 
# #### Task 6: If there is decline inform key stakeholders about the KPIs. If, there is a continous increse or steady sales...Continue the activities

# In[188]:


#### Step 1: importing the dependent packages


# In[189]:


import pandas as pd
import numpy as np
import statistics as sts
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as warning
warning.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# In[190]:


#### Step 2: Read the base dataset


# In[191]:


base_data_set =pd.read_csv("Rossmanntrain_new.csv").sample(1000)


# In[192]:


base_data_set['Date'].min()
base_data_set.count()


# In[193]:


base_data_set['Date'].max()


# In[ ]:





# In[194]:


## Subset the data for the first three months 
base_data_set=base_data_set[(base_data_set['Date'] >= '2013-01-01') & (base_data_set['Date'] <= '2013-03-31')]


# In[195]:


base_data_set['Date'].min()


# In[196]:


base_data_set['Date'].max()


# In[197]:


base_data_set.count()


# In[198]:


base_data_set.describe().columns


# ### Preprocessing Steps: 
# 
# #####  1: Read the base dataset
# ##### 2: Null value treatement 
# ##### 3: Outlier Treatment
# #####  4: Encoded variables and Dummy varaibles
# ##### 5: Identify the important columns
# ##### 6: Univaraite Analysis 
# ##### 7: Bivariate Analysis
# ##### 8 : OverSampling & Under Sampling
# 
# ### Additional Step :  
# 
# ##### 1: Garbage Value Treatement
# ##### 2: Normalization
# ##### 3: Standardization 

# In[199]:


#3.Null value treatement


# In[200]:


def nullValueAnalysis(Base_DataSet):
# find the percentage of null values in the base dataset 
     null_values=(Base_DataSet.isna().sum()/Base_DataSet.shape[0])*100 
# make a dataframe for this null value percentage 
     null_values=pd.DataFrame(null_values)
# Add null value percentage column to existing data Frame
     null_values.columns=['Null_Value %']
# sorting the Null value percentage column by descending order
     null_values=null_values['Null_Value %'].sort_values(ascending=False)
# make a dataframe for this null value percentage column
     null_values=pd.DataFrame(null_values)
# take only less than 30 % of the null values to prepare the analytical data set 
     Base_DataSet.drop(null_values[null_values['Null_Value %'] > 30].index,axis= 1,inplace=True) 
     for i in Base_DataSet.describe().columns:
        Base_DataSet[i].fillna(Base_DataSet[i].median(),inplace=True)
       # for i in cust_Base_dataSet.describe().columns:
       #     cust_Base_dataSet[i].fillna(cust_Base_dataSet[i].median(),inplace=True)
 
     return Base_DataSet    


# In[201]:


Base_DataSet=nullValueAnalysis(base_data_set)


# In[202]:


# 3. Outlier Treatment


# In[203]:


def outliersAnalysis(base_data_set):
    for i in base_data_set.describe().columns:
        x=np.array(base_data_set[i])
        p=[]
        Q1 = base_data_set[i].quantile(0.25)
        Q3 = base_data_set[i].quantile(0.75)
        IQR = Q3 - Q1
        LTV= Q1 - (1.5 * IQR)
        UTV= Q3 + (1.5 * IQR)
        for j in x:
            if j <= LTV or j>=UTV:
                p.append(sts.median(x))
            else:
                p.append(j)
        base_data_set[i]=p
    return base_data_set


# In[204]:


Base_DataSet=outliersAnalysis(base_data_set)


# In[205]:


def univariate_analysis(base_null_value_treated):
    import matplotlib.pyplot as plt
    col=[]
    for i in base_null_value_treated.describe().columns:
        var=base_null_value_treated[i].value_counts().values.var()
        col.append([i,var])
        variance_table=pd.DataFrame(col)
        variance_table[variance_table[1]>100][0].values
    return variance_table[variance_table[1]>100][0].values
        


# In[206]:


#Base_DataSet =univariate_analysis(Base_DataSet)


# In[207]:


# find the correlation between the one and another columns and take only gratethan 0.5 correlation and less than -0.5 correlation
def bivariate_analysis(Base_dataSet):
    correlation=[]
    for i in Base_dataSet.describe().corr().columns:
        for j in Base_dataSet.describe().corr().columns:
            if i != j :
                corr=Base_dataSet[[i,j]].corr().values[1][0]
                d= {
                    'Column name1':i,
                    'Column name2':j,
                    'corr':corr
                   }
                correlation.append(d)
    
    correlation_table=pd.DataFrame(correlation)
    
    variables_bivariate=correlation_table[(correlation_table['corr']>0.5) | (correlation_table['corr']<-0.5)]['Column name1'].values

    return variables_bivariate  


# In[208]:


# Base_DataSet =bivariate_anaysis(Base_DataSet)


# In[209]:


# Normalization
Base_dataSet =Base_DataSet


# In[210]:


Base_dataSet.drop(['Date','StateHoliday'],axis=1,inplace=True)
Base_dataSet.dtypes


# In[211]:


Base_dataSet.dtypes
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(Base_dataSet)
minmax=pd.DataFrame(scaler.transform(Base_dataSet))
plt.figure(figsize=(10,10))
sns.heatmap(minmax.corr(),annot=True,cmap='viridis')


# In[ ]:





# In[212]:


# elbow method to find the number of cluster
def elbow(Base_DataSet,numberCluset):
    x=[]
    for i in range(1,numberCluset):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(Base_DataSet)
        x.append(kmeans.inertia_)
    plt.plot(range(1,numberCluset), x)
    plt.title('The elbow method')
    plt.xlabel('The number of clusters')
    plt.ylabel('Variance')
    plt.show()


# In[213]:


Base_DataSet =elbow(Base_dataSet,30)


# In[214]:


# K Means clustering
kmeans=KMeans(n_clusters=5)
kmeans.fit(Base_dataSet)
Base_dataSet['cluster']=kmeans.labels_
Base_dataSet.describe().columns.index
Base_dataSet


# In[ ]:


basedataframe= Base_dataSet[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','SchoolHoliday']]


# In[215]:


Base_dataSet[Base_dataSet.var().sort_values(ascending=False).head(10).index].sample(100):for i in basedataframe.describe().columns:
    print("***********",i,"**********")
    print(Base_dataSet.groupby('cluster').describe()[i])


# In[216]:



Base_dataSet[Base_dataSet.var().sort_values(ascending=False).head(10).index]


# #### Visualizations 

# In[217]:


for i in Base_dataSet[Base_dataSet.var().sort_values(ascending=False).head(10).index]:
    for j in Base_dataSet[Base_dataSet.var().sort_values(ascending=False).head(10).index]:
        if i!=j:
            plt.scatter(Base_dataSet[Base_dataSet['cluster']==1][[i]],Base_dataSet[Base_dataSet['cluster']==1][[j]])
            plt.scatter(Base_dataSet[Base_dataSet['cluster']==2][[i]],Base_dataSet[Base_dataSet['cluster']==2][[j]])
            plt.scatter(Base_dataSet[Base_dataSet['cluster']==3][[i]],Base_dataSet[Base_dataSet['cluster']==3][[j]])
            plt.scatter(Base_dataSet[Base_dataSet['cluster']==4][[i]],Base_dataSet[Base_dataSet['cluster']==4][[j]])
            plt.scatter(Base_dataSet[Base_dataSet['cluster']==0][[i]],Base_dataSet[Base_dataSet['cluster']==0][[j]])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()


# In[218]:


plt.scatter(Base_dataSet[Base_dataSet['cluster']==1]['Sales'],Base_dataSet[Base_dataSet['cluster']==1]['Customers'])
plt.scatter(Base_dataSet[Base_dataSet['cluster']==2]['Sales'],Base_dataSet[Base_dataSet['cluster']==2]['Customers'])
plt.scatter(Base_dataSet[Base_dataSet['cluster']==3]['Sales'],Base_dataSet[Base_dataSet['cluster']==3]['Customers'])
plt.scatter(Base_dataSet[Base_dataSet['cluster']==4]['Sales'],Base_dataSet[Base_dataSet['cluster']==4]['Customers'])
plt.scatter(Base_dataSet[Base_dataSet['cluster']==5]['Sales'],Base_dataSet[Base_dataSet['cluster']==5]['Customers'])


# In[ ]:




