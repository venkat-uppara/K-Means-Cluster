#  K-Means-Cluster Interview Question 

#### 1. What is the draw back of K-Means Algorithm ?
        1.Cluser centriod  has to be specified which is basically not known in prior.
        2.Selection of intial cluser centriod is difficult.
        3.Different intial centroids results in different clusters.
        4.May run infinitely if the stopping criteria is not satisfied.
        5.Rescaling your datasets(normalizaHon or standardizaHon) will completely change results. While this itself is not bad,not realizing     that you have to spend extra attention to scaling  your data.
        6. Outliers can change your cluster in K means to very large extent
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
#### 4. How to Change the Outlier in the data .Does K-Means affect because of the Outlier ?
#### 5. How to proceed the when you have too many null values in the data ?
#### 6. How to select the right number of cluster During based on the data ?
#### 7. Is It right to always take the sample of the data ? what scenarios the clustering output might changes with respective to samples ?


## Problem Statment :
  ## The Data is avaliable for 3 years ie 12 Quarters .Please use the fourth month data to create a cluster and track the progress of the cluster across the months 
#### Task 1: Subset the data for the first three months
#### Task 2: K Elbow ( on first 3 months data )
#### Task 3: K Means ( Create Clusters )
#### Task 4: Visualize the clusters ( Create Clusters )(Sample of observations)
#### Task 5: The clusters formed in the step 3 needs to be tracked across next 30 Quarters
#### Task 6: If there is decline inform key stakeholders about the KPIs. If, there is a continous increse or steady sales...Continue the activities
