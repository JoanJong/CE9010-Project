# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:03:33 2019

@author: yuhui
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data=pd.DataFrame.as_matrix(pd.read_csv('foo.csv'))
r,c=data.shape
r*=0.8
#np.random.shuffle(data)
#np.savetxt("hoo.csv", data, delimiter=",")
train_data=data[:int(r),:]
X1=train_data[:,0]
X2=train_data[:,1]
Y_train=train_data[:,2]
test_data=data[int(r):,:]
y_test=test_data[:,3]

k=2
plt.scatter(X1, X2, label='True position')
kmeans = KMeans(n_clusters=k)  
kmeans.fit(train_data[:,0:2])
y_pred=kmeans.fit_predict(train_data[:,0:2])
plt.scatter(X1,X2, c=kmeans.labels_, cmap='rainbow')
plt.colorbar()
#plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')



#Validation
y_score=kmeans.predict(test_data[:,0:2])




#Precision recall
#y_test=OG training data
#y_score=predicted

y_test = y_test[:,None]
y_score = y_score[:,None]
testr,testc=y_test.shape

TP=0
FP=0
FN=0
TN=0

for i in range(testr):
    if y_test[i][0] == 1 and y_score[i][0] == 1:
        TP+=1
    elif y_test[i][0] == 0 and y_score[i][0] == 1:
        FP+=1
    elif y_test[i][0] == 1 and y_score[i][0] == 0:
        FN+=1
    else:
        TN+=1

print(TP/(TP+FP))
print(TP/(TP+FN))


