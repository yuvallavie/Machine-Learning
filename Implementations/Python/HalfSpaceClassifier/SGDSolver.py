import numpy as np

def Hinge(X_train,Y_train,W):
    loss = 0;
    m = len(X_train);
    for i in range(m):
        loss += np.maximum(0,1 - Y_train[i] * g(X_train[i],W));
    return loss/m;
  
# Helpers for surrogate optimization
def g(x,W):
    return (np.inner(x,W));

def Derivative(y,x,W,i):
    condition = 1 - y * g(x,W);
    if(condition <= 0):
        return 0;
    else:
        return -y*x[i];
      

#%%
# ------------------------------------------------------------------------
#                               Batch Gradient Descent
# ------------------------------------------------------------------------
def SGD(X,Y,objective,derivative,iterations,lRate,bSize):
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    m = len(X[0]);
    W = np.zeros(m);
    for i in range(iterations):
        # Print the function
        currLoss = objective(X,Y,W);
#        print("Objective:", currLoss);
        if (currLoss == 0):
               print("Gradient Descent optimization terminated after " + str(i) + " runs");
               return W;
        
        # Initialize all gradients each run
        subgradientW = np.zeros(m); 
        # Batch Gradient Descent (Not sure if this is the stochastic equivalent)
        # calculate gradients for each sample point in the batch
        indices = np.random.randint(low=0,high=len(X),size=bSize);
        for j in indices:
            for k in range(m):
                subgradientW[k] += derivative(Y[j],X[j],W,k);
        # Add the gradients to the weights    
        W -= lRate * subgradientW;
        # Lower the learning rate
        lRate = 0.99 * lRate;

    print("Gradient Descent optimization Complete\nIterations: ",i);    
    return W;

def Solve(X,Y,maxIter=1000,lRate=0.01,bSize=10):
    return SGD(X,Y,Hinge,Derivative,maxIter,lRate,bSize);