#!/usr/bin/env python
# coding: utf-8

# In[154]:


# import pandas package 
import pandas as pd
import numpy as np
import statistics as sts
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as warning
warning.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# In[155]:


# Read CSV file 
cust_Base_dataSet=pd.read_csv("application_train.csv")


# In[156]:


cust_Base_dataSet.shape


# In[157]:


# take sample 100 rows to prepare the analytical data set then finally replace this sample analytical dataset 
# to base analytical dataset
cust_Base_dataSet=cust_Base_dataSet.sample(100)


# In[158]:


cust_Base_dataSet.head(5)


# ### Preprocessing steps
#   1. Null value Analysis
#   2. Outlier Analysis
#   3. Univariate Analysis
#   4. BiVariates Analysis
#   5. Normalization
#    
# 
# 

# ### 1.Null Values Analysis

# In[159]:


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
     Base_DataSet.drop(null_values[null_values['Null_Value %'] < 30].index,axis= 1,inplace=True) 
     for i in cust_Base_dataSet.describe().columns:
        cust_Base_dataSet[i].fillna(cust_Base_dataSet[i].median(),inplace=True)
        for i in cust_Base_dataSet.describe().columns:
            cust_Base_dataSet[i].fillna(cust_Base_dataSet[i].median(),inplace=True)
 
     return Base_DataSet    


# In[160]:


cust_Base_dataSet=nullValueAnalysis(cust_Base_dataSet)


# ### 2.Outlier Analysis

# In[20]:


# Outlier formula Q1=0.25 % of base data set
#                 Q3=0.75 % of base data set
#                 IQR(Inter Quartile range)=Q3-Q1
#                 LTV(lower Threshold value)=Q1-(1.5*IQR)
#                 UTV(Upper Threshold value)=Q3+(1.5*IQR)
# if data sets below LTU or above UTV then take median for data sets
def Oulier_Analysis(Base_dataSet):
    for i in Base_dataSet.describe().columns:
        x=np.array(Base_dataSet[i])
        p=[]
        Q1=Base_dataSet[i].quantile(0.25)
        Q3=Base_dataSet[i].quantile(0.75)
        IQR=Q3-Q1
        LTV=Q1-(1.5 * IQR)
        UTV=Q3+(1.5 * IQR)
        for j in x:
            if j <=LTV or j >= UTV:
                p.append(sts.median(x))
            else :
                 p.append(j)
        Base_dataSet[i]=p         

        return Base_dataSet 


# In[21]:


cust_Base_dataSet=Oulier_Analysis(cust_Base_dataSet)


# ### 3. Univariate Analysis
# Collecting and analysing the data on one varaiable 
# for more information please check the following link
# https://www.geeksforgeeks.org/univariate-bivariate-and-multivariate-data-and-its-analysis/

# In[22]:


# find the varience for each column in the given dataset

def univariate_analysis(Base_dataSet):
    col=[]
    for i in Base_dataSet.describe().columns:
        var=Base_dataSet[i].value_counts().values.var()
        col.append([i,var])
        var_Table= pd.DataFrame(col)
        var_Table[var_Table[1]>100][0].values
        
    return var_Table[var_Table[1]>100][0].values    
   


# In[23]:


cust_Base_dataSet_uni=univariate_analysis(cust_Base_dataSet)


# ### 4.Bivariate Analysis

# In[24]:


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
            
        
        


# In[ ]:





# In[25]:


bivariate_variables=bivariate_analysis(cust_Base_dataSet)


# In[14]:


def bivariate_analysis(bivariate_variables):
    for i in bivariate_variables:
        for j in bivariate_variables:
            if i!=j:
                sns.jointplot(cust_Base_dataSet[i],cust_Base_dataSet[j])


# In[ ]:


bivariate_analysis(bivariate_variables)


# ### 5.Normalization

# In[161]:


for i in cust_Base_dataSet.describe().columns:
       cust_Base_dataSet[i].fillna(cust_Base_dataSet[i].median(),inplace=True)
       
       cust_Base_data_con=cust_Base_dataSet   
   


# In[162]:


cust_Base_data_con.drop(['FONDKAPREMONT_MODE',
'HOUSETYPE_MODE',
'TOTALAREA_MODE',
'WALLSMATERIAL_MODE',
'EMERGENCYSTATE_MODE'],axis=1,inplace=True)


# In[163]:


scaler=MinMaxScaler()


# In[164]:


scaler.fit(cust_Base_data_con)
minmax=pd.DataFrame(scaler.transform(cust_Base_data_con))
minmax.columns=cust_Base_data_con.columns
minmax.head(10)


# #### Find the correlation betweeen the data column

# In[165]:


plt.figure(figsize=(30,30))
sns.heatmap(cust_Base_dataSet.corr(),annot=True,cmap='viridis')


# #### Using the elbow method to find the optimal number of clusters
# 

# In[166]:


def elbow(Analytical_dataSet,numberCluset):
    x=[]
    for i in range(1,numberCluset):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(Analytical_dataSet)
        x.append(kmeans.inertia_)
    plt.plot(range(1,numberCluset), x)
    plt.title('The elbow method')
    plt.xlabel('The number of clusters')
    plt.ylabel('Variance')
    plt.show()


# In[167]:


elbow(cust_Base_dataSet,30)


# #### K Means clustering

# In[168]:


Analytical_dataSet=minmax


# In[169]:


kmeans=KMeans(n_clusters=5)
kmeans.fit(Analytical_dataSet)
len(kmeans.labels_)
kmeans.labels_


# In[170]:


Analytical_dataSet['cluster']=kmeans.labels_


# In[171]:


Analytical_dataSet['cluster'].value_counts()


# In[172]:


Analytical_dataSet.head(10)


# In[173]:


for i in Analytical_dataSet.describe().columns:
    print("***********",i,"**********")
    print(Analytical_dataSet.groupby('cluster').describe()[i])


# #### Visualizations 

# In[ ]:


Analytical_dataSet.columns


# In[ ]:


plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==0][['APARTMENTS_AVG']],Analytical_dataSet[Analytical_dataSet['cluster']==0][['FLOORSMAX_AVG']])
plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==1][['APARTMENTS_AVG']],Analytical_dataSet[Analytical_dataSet['cluster']==1][['FLOORSMAX_AVG']])
plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==2][['APARTMENTS_AVG']],Analytical_dataSet[Analytical_dataSet['cluster']==2][['FLOORSMAX_AVG']])
plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==3][['APARTMENTS_AVG']],Analytical_dataSet[Analytical_dataSet['cluster']==3][['FLOORSMAX_AVG']])
plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==4][['APARTMENTS_AVG']],Analytical_dataSet[Analytical_dataSet['cluster']==4][['FLOORSMAX_AVG']])















# In[ ]:


for i in Analytical_dataSet[Analytical_dataSet.var().sort_values(ascending=False).head(10).index].sample(100):
    for j in Analytical_dataSet[Analytical_dataSet.var().sort_values(ascending=False).head(10).index].sample(100):
        if i!=j:
            plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==1][[i]],Analytical_dataSet[Analytical_dataSet['cluster']==1][[j]])
            plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==2][[i]],Analytical_dataSet[Analytical_dataSet['cluster']==2][[j]])
            plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==3][[i]],Analytical_dataSet[Analytical_dataSet['cluster']==3][[j]])
            plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==4][[i]],Analytical_dataSet[Analytical_dataSet['cluster']==4][[j]])
            plt.scatter(Analytical_dataSet[Analytical_dataSet['cluster']==0][[i]],Analytical_dataSet[Analytical_dataSet['cluster']==0][[j]])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()


# ##### Agglomerative clustering

# In[ ]:


# Using the dendrogram to find the optimal number of clusters


# In[ ]:


import scipy.cluster.hierarchy as sch
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering


# In[ ]:


dendrogram = sch.dendrogram(sch.linkage(Analytical_dataSet, method = 'single'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[184]:


agglomerative=AgglomerativeClustering(n_clusters=5,affinity = 'euclidean', linkage = 'single')
y_hc=agglomerative.fit_predict(Analytical_dataSet)
X = Analytical_dataSet.iloc[:, [0, 1]].values


# In[185]:


# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('COMMONAREA_MEDI')
plt.ylabel('YEARS_BUILD_MEDI ')
plt.legend()
plt.show()


# In[ ]:




