# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Pick customer segment quantity (k)

2.Seed cluster centers with random data points.

3.Assign customers to closest centers.

4.Re-center clusters and repeat until stable.


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: BAKKIYA LAKSHMI M
RegisterNumber:  2222220006
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
x

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.show()

k=5
kmeans = KMeans(n_clusters=k)
kmeans.fit(x)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroid:")
print(centroids)
print("Labels:")
print(labels)

colors =['r','g','b','c','m']
for i in range(k):
  cluster_points =x[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],color=colors[i], label = f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
  
plt.scatter(centroids[:,0], centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('k-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show
```

## Output:
DATA:

![322957576-8971b362-90f4-4212-9bb7-bde7c21947fc](https://github.com/Bakkiyalakshmi29/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119406233/95e0f91e-2cd4-4f34-b944-14026e996a9b)

![322957664-41d64d86-fed3-466e-9711-59dc813e47cf](https://github.com/Bakkiyalakshmi29/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119406233/ab2e1df2-1c35-405f-a066-284fa7a8d60c)

SCATTER PLOT:

![322957826-7439bbac-90f6-40a3-afce-0ed387b2a478](https://github.com/Bakkiyalakshmi29/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119406233/3523b85c-0a64-4daf-97ff-90d2619d1725)

CENTROIDS:

![322957987-8968a68e-0e90-4758-a31b-e445705b17b1](https://github.com/Bakkiyalakshmi29/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119406233/370a653e-c215-4a24-9193-6d2c33060e93)

KMeans Clustering:

![322958209-1416f4fd-38d2-411d-b1e8-fa191d8f1b1e](https://github.com/Bakkiyalakshmi29/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119406233/52dcf808-7e35-4a61-a5ce-6ed869b3c37f)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
