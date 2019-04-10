#%%
# Import libraries

# math library
import numpy as np

# visualization library
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png2x','pdf')
import matplotlib.pyplot as plt

# machine learning library
from sklearn.linear_model import LogisticRegression

# 3d visualization
from mpl_toolkits.mplot3d import axes3d

# computational time
import time
import os
os.chdir('C:/Users/My/Desktop')
#%%
# import data with numpy

data = np.loadtxt('all_out_mat.csv', delimiter=',')

# number of training data
n = data.shape[0] #YOUR CODE HERE
print('Number of training data=',n)

# print
print(data[:10,:])
print(data.shape)
print(data.dtype)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
#%%
x1 = train[:,0] # bad
x2 = train[:,1] # good
x1.shape
#%%

idx_good= (train[:,3]==1) 
idx_bad = (train[:,3]==0) 

plt.figure(1,figsize=(6,6))
plt.scatter(x1[idx_good], x2[idx_good], s=60, c='r', marker='+', linewidths=2, label='Admitted') #YOUR CODE HERE
plt.scatter(x1[idx_bad], x2[idx_bad], s=60, c='g', marker='o', linewidths=2, label='Rejected') #YOUR CODE HERE
plt.title('Training data')
plt.xlabel('Exam grade 1')
plt.ylabel('Exam grade 2')
plt.legend()
plt.show()
#%%
def sigmoid(z):
    sigmoid_f = 1 / (1 + np.exp(-z)) #YOUR CODE HERE
    return sigmoid_f 



#%%
# construct the data matrix X
n = data.shape[0]
X = np.ones([n,3]) 
X[:,1:3] = data[:,0:2]
print(X.shape)
print(X[:5,:])


# parameters vector
w = np.array([0,0,0])[:,None] # [:,None] adds a singleton dimension
print(w.shape)


# predictive function definition
def f_pred(X,w): 
    p = sigmoid(X.dot(w)) #YOUR CODE HERE
    return p


# Test predicitive function 
y_pred = f_pred(X,w)
print(y_pred[:3])
#%%
# loss function definition
def loss_logreg(y_pred,y): 
    n = len(y)
    loss = -1/n* ( y.T.dot(np.log(y_pred)) + (1-y).T.dot(np.log(1-y_pred)) ) #YOUR CODE HERE
    return loss


# Test loss function 
y = data[:,3][:,None] # label 
print(y.shape)
#print(y)
y_pred = f_pred(X,w) # prediction
loss = loss_logreg(y_pred,y)
print(loss)
#%%
# gradient function definition
def grad_loss(y_pred,y,X):
    n = len(y)
    grad = 2/n* X.T.dot(y_pred-y) #YOUR CODE HERE
    return grad


# Test grad function 
y_pred = f_pred(X,w)
grad = grad_loss(y_pred,y,X)
print(grad)    
#%%
# gradient descent function definition
def grad_desc(X, y , w_init=np.array([0,0,0])[:,None] ,tau=1e-4, max_iter=5000):

    L_iters = np.zeros([max_iter]) # record the loss values
    w_iters = np.zeros([max_iter,2]) # record the loss values
    w = w_init # initialization
    for i in range(max_iter): # loop over the iterations
        y_pred = f_pred(X,w) # linear predicition function  #YOUR CODE HERE
        grad_f = grad_loss(y_pred,y,X) # gradient of the loss #YOUR CODE HERE
        w = w - tau* grad_f # update rule of gradient descent #YOUR CODE HERE
        L_iters[i] = loss_logreg(y_pred,y) # save the current loss value 
        w_iters[i,:] = w[0],w[1] # save the current w value 
        
    return w, L_iters, w_iters


# run gradient descent algorithm
start = time.time()
w_init = np.array([0,0,0])[:,None]
#w_init = np.array([0,0,0])[:,None]
tau = 1e-4; max_iter = 5000
w, L_iters, w_iters = grad_desc(X,y,w_init,tau,max_iter)
print('Time=',time.time() - start)
print(L_iters[-1])
print(w)


# plot
plt.figure(3)
plt.plot(np.array(range(max_iter)), L_iters)
plt.xlabel('Iterations')
plt.ylabel('Loss value')
plt.show()
#%%
# compute values p(x) for multiple data points x
x1_min, x1_max = X[:,1].min(), X[:,1].max() # min and max of grade 1
x2_min, x2_max = X[:,2].min(), X[:,2].max() # min and max of grade 2
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max)) # create meshgrid
X2 = np.ones([np.prod(xx1.shape),3]) 
X2[:,1] = xx1.reshape(-1)
X2[:,2] = xx2.reshape(-1)
p = f_pred(X2,w)
p = p.reshape(xx1.shape)


# plot
plt.figure(4,figsize=(6,6))
plt.scatter(x1[idx_good], x2[idx_good], s=60, c='r', marker='+', linewidths=2, label='Admitted') #YOUR CODE HERE
plt.scatter(x1[idx_bad], x2[idx_bad], s=60, c='b', marker='o', linewidths=2, label='Rejected') #YOUR CODE HERE
plt.contour(xx1, xx2, p, [0.5], linewidths=2, colors='k') #YOUR CODE HERE
plt.xlabel('bad words')
plt.ylabel('good words')
plt.legend()
plt.title('Decision boundary')
plt.show()


# record p values
p_gd = p
