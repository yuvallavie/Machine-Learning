# ---------------------------------------------------------------------
#                       Stochastic Gradient Descent
# ---------------------------------------------------------------------
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
#%%
# ---------------------------------------------------------------------
#                       Hinge Loss
# ---------------------------------------------------------------------
# Surrogate Loss
def HingeObjective(X_train,Y_train,W,b):
    loss = 0;
    m = len(X_train);
    for i in range(m):
        loss += np.maximum(0,1 - Y_train[i] * g(X_train[i],W,b));
    return loss/m;
  
# Helpers for surrogate optimization
def g(x,W,b):
    return (np.inner(x,W) + b);

def HingeDerivativeW(y,x,W,b,i):
    condition = 1 - y * g(x,W,b);
    if(condition <= 0):
        return 0;
    else:
        return -y*x[i];
      
def HingeDerivativeB(y,x,W,b):
    condition = 1 - y * g(x,W,b);
    if(condition <= 0):
        return 0;
    else:
        return -y;
#%%
# ------------------------------------------------------------------------
#                               Batch Gradient Descent
# ------------------------------------------------------------------------
def SGD(X,Y,objective,derivatives,iterations,lRate,bSize):
    # Initial Guess
    m = len(X[0]);
    W = np.zeros(m);
    b = 0;
    for i in range(iterations):
        # Print the function
        currLoss = objective(X,Y,W,b);
#        print("Objective:", currLoss);
        if (currLoss == 0):
               print("Gradient Descent optimization Complete after " + str(i) + " runs");
               return [W,b];
        
        # Initialize all gradients each run
        subgradientW = np.zeros(m);
        subgradientB = 0 
        # Batch Gradient Descent (Not sure if this is the stochastic equivalent)
        # calculate gradients for each sample point in the batch
        indices = np.random.randint(low=0,high=len(X),size=bSize);
        for j in indices:
            subgradientB += derivatives[0](Y[j],X[j],W,b);
            for k in range(m):
                subgradientW[k] += derivatives[1](Y[j],X[j],W,b,k);
        # Add the gradients to the weights    
        b -= lRate * subgradientB;
        W -= lRate * subgradientW;
        # Lower the learning rate
        lRate = 0.99 * lRate;

    print("Gradient Descent optimization Complete");    
    return [W,b];

#%%
def HingeLoss(X,Y,maxIter=1000,lRate=0.01,bSize=10):
    return SGD(X,Y,HingeObjective,[HingeDerivativeB,HingeDerivativeW],maxIter,lRate,bSize);

def VisualizeResult2D(W,b,X,Y):
    for i in range(len(X)):
        if(Y[i] == -1):
            plt.scatter(X[i][0],X[i][1],color='r');
        else:
            plt.scatter(X[i][0],X[i][1],color='b');
            
    ls = np.linspace(np.min(X) - 0.1,np.max(X) + 0.1);
    plt.plot(ls,np.inner(-(W[0]/W[1]),ls) - b/W[1])
    plt.show()
    