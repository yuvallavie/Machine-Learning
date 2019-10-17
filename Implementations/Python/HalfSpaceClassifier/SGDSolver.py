import numpy as np

# Helpers for surrogate optimization
def g(X,W):
    return (np.inner(X,W));
    
# Loss function

def HingeLoss(X,W,y):
    return np.maximum(0,1 - y*g(X,W))
    
def HingeDerivative(Y,X,W):
    condition = 1 - Y * g(X,W);
    if(condition > 0):
        return -Y*X;
    return 0;
    
def HingeCost(X_train,Y_train,W):
    sum = 0;
    m = len(X_train);
    for i in range(m):
        #loss += np.maximum(0,1 - Y_train[i] * g(X_train[i],W));
        sum += HingeLoss(X_train[i],W,Y_train[i]);
    return sum/m;    

#%%
# ------------------------------------------------------------------------
#                       Stochastic Gradient Descent
#                       batchSize = 1 -> Incremental Mode
#                       btachSize = len(X) -> Gradient Descent
# ------------------------------------------------------------------------
def SGD(X,Y,Obj,Der,T,lRate,batchSize):
    # Add the psuedo-feature 1 for the bias calculation
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    m = len(X[0]);
    # Initialize the weights
    W = np.zeros((T,m));
    for t in range(T):
        # Batch mode (use 1 for incremental, use len(X) for deterministic gradient descent)
        batch = np.random.randint(-1,len(X),batchSize);
        for j in batch:   
            grad = Der(Y[j],X[j],W[t-1]);
            W[t] = W[t-1] - lRate * grad;
    return np.mean(W,axis=0);

def Solve(X,Y,maxIter=1000,lRate=0.01,bSize=10):
    return SGD(X,Y,HingeCost,HingeDerivative,maxIter,lRate,bSize);  