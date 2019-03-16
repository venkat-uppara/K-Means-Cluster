#  K-Means-Cluster Interview Question 

#### 1. What is the draw back of K-Means Algorithm ?
        1.Cluser centriod  has to be specified which is basically not known in prior.
        2.Selection of intial cluser centriod is difficult.
        3.Different intial centroids results in different clusters.
        4.May run infinitely if the stopping criteria is not satisfied.
        5.Rescaling your datasets(normalizaHon or standardizaHon) will completely change results. While this itself is not bad,not realizing     that you have to spend extra attention to scaling  your data.
        6. Outliers can change your cluster in K means to very large extent
        7. This algorithm is noty handle if the data set have categorical 
#### 2. Explain K-Mean Algorithm with taking 5 rows and 2 columns as examples ?
        K-Means Algorithm
             Step 1: K-Means algorithm works by randomly placing the K centriods ,one for each cluster
             Step 2. Assign each data points it is closest to the centriod. Creating a group is used by Euclidian distence formula to measeure    the distence from data points to centriod. 
             Step 3. once data points has been classified to a group, recaluculate position of the centriod. the new centriod position is determined by mean of the  all the data points in the group.
             Step 4: Repeate the step 2 and 3 ,involving the assignment of points to centriod and moving the centriod until the centriod no longer to move.
             
             Ex: 5 rows and 2 columns Data set
             
             1. Specify the desired number of clusters K : Let us choose k=2 for these 5 data points. 
             2. Randomly assign each data point to a cluster 
             3. Compute cluster centroids
             4. Re-assign each point to the closest cluster centroid
             5. Re-compute cluster centroids : Now, re-computing the centroids for both the clusters.
             6. Repeat steps 4 and 5 until no improvements are possible.When there will be no further switching of data points between two clusters for two successive repeats. It will mark the termination of the algorithm if not explicitly mentioned
             
                      
#### 3. Does K-Means affected by Null Values and How to Treat the null values ?
         
         Ans : No Null values will not affected the  K-Means algorithm because it is caluclute the distence between centriod and data points . but its a better to handle the null values. first we need to check the percentage of null values present in the data set and better way to avoid if null value percentage is greater than 30 %. we can replace null values by  median for continious variables and for categorical variable replace with most predominate values in the given feature.
         
         Ex: def nullValueAnalysis(Base_DataSet):
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
              # Continous varable     
                  for i in Base_DataSet.describe().columns:
                      Base_DataSet[i].fillna(Base_DataSet[i].median(),inplace=True)
               # Categorical  varable            
                        for i in cust_Base_dataSet.describe(include="object").columns:
                          cust_Base_dataSet[i].fillna(cust_Base_dataSet[i].median(),inplace=True)
 
                     return Base_DataSet 
           
         
#### 4. How to Change the Outlier in the data .Does K-Means affect because of the Outlier ?
       Outliers can change your cluster in K means to very large extent . for examples We want to find 2 clusters in the 5 data points  using K-means. However, say there's also a single outlier, located very far from either of the 'true' clusters. Maybe millions of times further away from any other point than any other points are to each other. If we chose the centroids to be the centers of the true clusters (the best 'representative' configuration), the value of the loss function would be very high. The loss function is the sum of squared distances from each point to its assigned cluster centroid. It would be high because the outlier is so far from the nearest centroid. Therefore, K-means would reduce the loss function by choosing the outlier itself to be one of centroids, and placing the other centroid somewhere in the middle of the remaining data.
        Below is the code snippet to handle the Outlier
        
        # Outlier formula Q1=0.25 % of base data set
        # Q3=0.75 % of base data set
        # IQR(Inter Quartile range)=Q3-Q1
        # LTV(lower Threshold value)=Q1-(1.5*IQR)
        # UTV(Upper Threshold value)=Q3+(1.5*IQR)
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
       

#### 5. How to proceed the when you have too many null values in the data ?
      first we need to check the percentage of null values present in the data set and better way to avoid if null value percentage is greater than 30%
                 # take only less than 30 % of the null values to prepare the analytical data set 
                  Base_DataSet.drop(null_values[null_values['Null_Value %'] > 30].index,axis= 1,inplace=True) 

#### 6. How to select the right number of cluster During based on the data ?
        It's based on the problem statement / given data set .if given data set have clasification featuers then we can easily choose the number of cluster. in real time most of the time client will  give the number of cluster other wise you need to use Using the elbow method to find the optimal number of clusters.
        Sample code snippet :
        
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
                 
#### 7. Is It right to always take the sample of the data ? what scenarios the clustering output might changes with respective to samples ?
        No it's not always to take the sample of the data. if your building a complex models then by using sample data it will reduce the cost and increase the accurecy of the model. if data set have  outliers then the cluster output might changes with respective to sample.

## Problem Statment :
  ## The Data is avaliable for 3 years ie 12 Quarters .Please use the fourth month data to create a cluster and track the progress of the cluster across the months 
#### Task 1: Subset the data for the first three months
#### Task 2: K Elbow ( on first 3 months data )
#### Task 3: K Means ( Create Clusters )
#### Task 4: Visualize the clusters ( Create Clusters )(Sample of observations)
#### Task 5: The clusters formed in the step 3 needs to be tracked across next 30 Quarters
#### Task 6: If there is decline inform key stakeholders about the KPIs. If, there is a continous increse or steady sales...Continue the activities
