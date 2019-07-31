# --------------------------------------------------------
#                   Half Space Classification
#                   Gradient Descent
# --------------------------------------------------------
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
#%%
# Functions
def DeterministicLabeler(X,Y):
    for i in range(M):
        if(X[i][0] + X[i][1] > 0.5):
            Y[i] = 1;
        else:
            Y[i] = -1;
#%%
# Create mock data
# Number of Samples (Instances)        
M = 10;
# Number of features
N = 2;
X = np.random.randn(M,N);
# Skeleton for labels
Y = np.zeros(M);
# The realizable labeling function f : X -> Y
DeterministicLabeler(X,Y)
#%%
# Visualizing the data to verify its the realizable case
for i in range(M):
    if(Y[i] == -1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.show()
#%%
# define our hypothesis class
# h in H <=> h = sign(<X,W> + b)
def LinearSeparator(w,X,b):
    return np.inner(w,X) + b;

def hypothesis(w,X,b):
    return np.sign(LinearSeparator(w,X,b));

def loss(y,yHat):
    if(y != yHat):
        return 1;
    return 0;

def lossReplacement(y,w,X):
    return y*np.inner(w,X);

def getLineFromLP(W,bias,X):
     return np.inner(-(W[0]/W[1]),X) - bias/W[1];

def SolveByLP(X,Y):
    A = np.ones(np.shape(X));
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = Y[i]*X[i][j]
    v = np.ones(len(A));
    u = np.zeros(len(W));
    # Linear Programming Solver
    c = W;
    b = v;
    from scipy.optimize import linprog
    res = linprog(-u, A_ub=A,b_ub=b)
    return res;
    

#%%
# Visualize everything
W = [0,0];
b = 1;
pred = hypothesis(W,X,b)
#%%
# Calculate the loss
Loss = 0;
for i in range(len(Y)):
    Loss += loss(Y[i],pred[i])
Loss /= len(Y);

#%%
result = SolveByLP(X,Y)
#%%
Xnew =  np.concatenate((X, np.ones((len(X),1))), axis=1);
A = np.ones(np.shape(Xnew));
for i in range(len(A)):
    for j in range(len(A[0])):
        A[i][j] = Y[i]*Xnew[i][j]
v = np.ones(len(A));
u = np.zeros(len(A[0]));
from scipy.optimize import linprog
res = linprog(-u, A_ub=A,b_ub=v)
#%%
# Visualize the result from linear programming
ls = np.linspace(np.min(X[:,0]),np.max(X[:,0]),100);
W = res.x
fig = plt.figure();
for i in range(M):
    if(Y[i] == -1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.plot(ls,getLineFromLP(W[0:-1],W[-1],ls),label="New",linestyle=':')
plt.show()
#%%
# Optimization Syntax
def objective(u,w):
    return np.inner(u,w);
def constraint(X,Y):
    A = np.ones(np.shape(Xnew));
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = Y[i]*Xnew[i][j]
    return A;
    