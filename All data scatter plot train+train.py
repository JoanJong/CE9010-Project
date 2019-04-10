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

Y_value=data[:,2]
X1=data[:,0]
X2=data[:,1]

plt.scatter(X1,X2, c=Y_value, cmap='rainbow')
plt.colorbar()
