#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import statistics as sts
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as warning
warning.filterwarnings("ignore")
home_credit_dataSet= pd.read_csv("application_train.csv")


# In[46]:


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
        #for i in Base_DataSet.describe(include='object').columns:
        #    Base_DataSet[i].fillna(Base_DataSet[i].median(),inplace=True)
 
     return Base_DataSet    


# In[47]:


home_credit_dataSet=nullValueAnalysis(home_credit_dataSet)


# In[36]:


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


# In[37]:


home_credit_dataSet=Oulier_Analysis(home_credit_dataSet)


# In[38]:


home_credit_dataSet


# In[39]:


for i in home_credit_dataSet.describe().columns:
        home_credit_dataSet[i].fillna(home_credit_dataSet[i].median(),inplace=True)
        
        home_credit_dataSet=home_credit_dataSet   


# In[40]:


home_credit_dataSet


# In[41]:


plt.figure(figsize=(30,30))
sns.heatmap(home_credit_dataSet.corr(),annot=True,cmap='viridis')


# In[44]:


home_credit_dataSet.drop('SK_ID_CURR',a)


# In[49]:


from sklearn.preprocessing import LabelEncoder
for i in home_credit_dataSet.describe(include='object').columns:
    home_credit_dataSet[i]=home_credit_dataSet[i].astype(str)
    le=LabelEncoder()
    le.fit(home_credit_dataSet[i])
    x=le.transform(home_credit_dataSet[i])
    home_credit_dataSet[i]=x


# In[51]:


home_credit_dataSet.columns


# In[53]:


y=home_credit_dataSet['AMT_ANNUITY']
x=home_credit_dataSet.drop('AMT_ANNUITY',axis=1)


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=53) 


# In[55]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[56]:


### Algorithm  Model Building
from sklearn.linear_model import LinearRegression
ln=LinearRegression()
ln.fit(X_train,y_train)
ln.predict(X_test)


# In[57]:


y_test.values


# In[58]:


#### Model Validation Steps


# In[59]:


ln.coef_


# In[60]:


coeff_table=pd.DataFrame(ln.coef_,X_train.columns)
coeff_table.reset_index(inplace=True)
coeff_table.columns=['column name','beta coeff']
coeff_table.sort_values('beta coeff',ascending=False).head()


# In[61]:


coeff_table.sort_values('beta coeff',ascending=False).tail()


# In[62]:


from sklearn.metrics import r2_score
r2_values=r2_score(y_test.values,ln.predict(X_test))


# In[64]:


from sklearn.metrics import mean_absolute_error,mean_squared_error

print("MAE",mean_absolute_error(y_test.values,ln.predict(X_test)))
print("MSE",mean_squared_error(y_test.values,ln.predict(X_test)))
print("RMSE",np.sqrt(mean_squared_error(y_test.values,ln.predict(X_test))))
print("MAPE",(((y_test-ln.predict(X_test))/y_test).sum()/X_test.shape[0])*100)
print("R2 Values",r2_values)


# In[65]:


from sklearn.model_selection import GridSearchCV


# In[66]:


params_grid = {"copy_X": [True, False],
               "fit_intercept": [True, False]}


# In[68]:


grid_search = GridSearchCV(ln,params_grid, scoring='accuracy')


# In[79]:


grid_search


# In[82]:


grid_search.fit(X_train,y_train)


# In[84]:


import statsmodels.api as sm
import statsmodels.formula.api as sfa 
model=sm.OLS(y_train,X_train) 
lm=model.fit()
lm.summary()


# In[ ]:





# In[ ]:




