# -----------------------------------------------------------------------
# Linear programming for binary classification in the realizable case
# -----------------------------------------------------------------------
#%%
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
#%%

def getLineFromSGD(W,bias,X):
     return np.inner(-(W[0]/W[1]),X) - bias/W[1];
#%%
# Create the seperable data
X = [(1,0.5),(0.3,0.4),(1,0.2),(0.1,0.4),(1.3,0.9),(1,1)];
Y = [-1,-1,-1,-1,1,1]
#%%
M = 10**1;
X = np.random.randn(M,2)
Y = np.zeros(M);
for i in range(M):
    if(X[i][0] + X[i][1] > 0.5):
        Y[i] = 1;
    else:
        Y[i] = -1;
#%%
# Visualize the data
for i in range(len(X)):
    if(Y[i] == 1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.title("Separable case")
plt.show()
#%%
# Preprocess, add b to the data
x_train = np.ones((len(X),len(X[0]) + 1))
for i in range(len(X)):
    x_train[i][0] = X[i][0];
    x_train[i][1] = X[i][1];
    
#%%
# maximize <u,w>
# s.t Aw >= b
#x_train = X
rows = len(x_train[0]);    
u = np.zeros(rows)
cols = len(x_train)
v = np.ones(cols)
# Create the matrix
A = np.zeros((cols,rows))
for i in range(cols):
    A[i] = np.inner(x_train[i],Y[i]);


#%%

#%%
# SCIPY
print("Maximize:" + str(u) + "T * W " + "\nSubject To: \n" + str(-A) + "w >" + str(np.transpose(-v)))
bounds = [];
for i in range(len(A[0])):
    bounds.append((None,None));
res = linprog(-u, A_ub = -A, b_ub = -v,bounds=bounds, options={"disp": True},method='simplex')
W = res.x
#%%
#%%
# Plot the separator
# Visualize the data
for i in range(len(X)):
    if(Y[i] == 1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.title("Separable case")
ls = np.linspace(-5,5);
plt.plot(ls,getLineFromSGD(W,W[2],ls))
plt.show()
#%%
# Verify Results
def predict(W,X):
    labels = [];
    for x in X:
        print(W,x)
        label = np.sign(np.inner(W,x));
        labels.append(label)
    return labels;
#%%
print(W)
yHat = predict(W,x_train)