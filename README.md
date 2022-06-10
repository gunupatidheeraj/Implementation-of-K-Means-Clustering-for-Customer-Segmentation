# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to work on.
3. From sklearn module import kmeans.
4. Define number of clusters to be made.
5. Assign the cluster values.
6. Plot the cluster using matplotlib.pyplot.
7. End the program.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Challa.Sandeep.
RegisterNumber: 212221240011.
*/
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init="k-means++")
  kmeans.fit(data.iloc[:,3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])

data["cluster"] =y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4= data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="blue",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="pink",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="green",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="red",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="orange",label="cluster4")
plt.title("Customer Segment")
plt.legend()
```

## Output:
![img1](https://user-images.githubusercontent.com/93427522/172998888-d8d50c99-e309-488f-9331-09609a6020bd.jpg)
![img2](https://user-images.githubusercontent.com/93427522/172998937-0d9847ff-c1bf-4d34-aad9-e0d5dc101aaf.jpg)
![img3](https://user-images.githubusercontent.com/93427522/172998949-ad7d15ef-60ad-47ce-8a9d-de7b8d86077f.jpg)
![img4](https://user-images.githubusercontent.com/93427522/172998962-3382c0e7-4738-4f90-aeed-988eaeddf9de.jpg)
![img5](https://user-images.githubusercontent.com/93427522/172998996-eb63e21c-32ed-4be8-a303-5b4767e33ef0.jpg)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
