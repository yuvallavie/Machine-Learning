# -------------------------------------------------------------------
#                       Stochastic Gradient Descent
#                       Binary Classifier
# -------------------------------------------------------------------
#%%
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
#%%
# Utilities
def getLineFromSGD(W,bias,X):
     return np.inner(-(W[0]/W[1]),X) - bias/W[1];
def visualizeResults(X,Y):
    for i in range(len(X)):
        if(Y[i] == -1):
            plt.scatter(X[i][0],X[i][1],color='r');
        else:
            plt.scatter(X[i][0],X[i][1],color='b');    
    
#%%
# Stochastic Gradient Descent
def SGD(X,Y,objective,derivative,iterations,lRate):
    # Initial Guess
    m = len(X[0]);
    W = np.zeros(m);
    b = 0;
    for i in range(iterations):
        # Print the function
        currLoss = objective(X,Y,W,b);
        print("Objective:", currLoss);
        if(currLoss == 0):
            return [W,b]
        
        
        # Initialize all gradients each run
        subgradientW = np.zeros(m);
        subgradientB = 0;
        # calculate gradients for each sample point
        for j in range(len(X)):
            subgradientB += DerivativeB(Y[j],X[j],W,b);
            for k in range(m):
                subgradientW[k] += derivative(Y[j],X[j],W,b,k);
   #             subgradientW[k] -= 0;
        # Add the gradients to the weights    
        b -= lRate * subgradientB;
        W -= lRate * subgradientW;

        
    return [W,b];
#%%
# Objective function to minimize
def f(x,W,b):
    return (np.inner(x,W) + b);

def Loss(x,y,W,b):
    return np.maximum(0,1 - y * f(x,W,b));

def DerivativeW(y,x,W,b,i):
    condition = 1 - y * f(x,W,b);
    if(condition <= 0):
        return 0;
    else:
        return -y*x[i];
    
def DerivativeB(y,x,W,b):
    condition = 1 - y * f(x,W,b);
    if(condition <= 0):
        return 0;
    else:
        return -y;
    
def Objective(X_train,Y_train,W,b):
    loss = 0;
    m = len(X_train);
    for i in range(m):
        loss += Loss(X_train[i],Y_train[i],W,b)
    return loss/m;

def fit(X,W,b):
    labels = [np.sign(f(W,x,b)) for x in X];
    return labels
    
def Loss01(y,yHat):
    if(y != yHat):
        return 1;
    return 0;
#%%
M = 10**2;
X = np.random.randn(M,2)
Y = np.zeros(M);
for i in range(M):
    res = np.random.binomial(1,0.95,1);
    if(X[i][0] - X[i][1] > -1):
        if(res == 1):
            Y[i] = 1;
        else:
            Y[i] = -1;
    else:
        if(res == 1):
            Y[i] = -1;
        else:
            Y[i] = 1;
        
#%%

#%%
# Test Zone
#X = np.array([(0.25,0.2),(0.4,0.05),(0.6,0.2),(0.5,0.5)])
#Y = np.array([-1,-1,1,1])
W = [0,0];
b = 0;
print("Loss before optimization: " ,Objective(X,Y,W,b))
#print(DerivativeW(X[1],W,Y[1],1))
[W,b] = SGD(X,Y,Objective,DerivativeW,1000,0.1); # Fix derivativeB too
print("Loss after optimization: " ,Objective(X,Y,W,b))
#%%
# Calculate the new loss
prediction = fit(X,W,b)
loss01 = 0;
for i in range(len(X)):
    loss01 += Loss01(Y[i],prediction[i])
print("Theoretical Loss:",loss01/len(X))
#%%
# Visualize the result
visualizeResults(X,Y)
ls = np.linspace(-3,3,100);
plt.plot(ls,getLineFromSGD(W,b,ls))
plt.show()
