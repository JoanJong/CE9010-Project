# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:33:18 2019

@author: yuhui
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


data=pd.DataFrame.as_matrix(pd.read_csv('foo.csv'))
r,c=data.shape
r*=0.8
train_data=data[:int(r),:]

Y_train=train_data[:,2]
test_data=data[int(r):,:]
X1=train_data[:,0]
X2=train_data[:,1]

plt.scatter(X1,X2, c=Y_train, cmap='rainbow')
plt.colorbar()
