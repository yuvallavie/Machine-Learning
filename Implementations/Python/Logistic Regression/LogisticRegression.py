import numpy as np

def sigmoid(x):
    ans = (1/(1+np.exp(-x)))
    if(ans == 1):
        return 0.99999;
    elif(ans == 0):
        return 0.00001;
    return ans;

def f(w,x):
    return np.inner(w,x);

def logloss(yHat,y):
    # Case y = 1
    A = -1 * (y * np.log(yHat))
    # Case y = 0
    B = -1 * ((1-y)*np.log(1-yHat));
    return A + B;

def logcost(yHat,y):
    m = len(y);
    count = 0;
    for i in range(m):
        count += logloss(yHat[i],y[i]);
    return count/m;

def derivative(yHat,y,x):
    return x*(yHat-y);

def predict(w,x):
    x = np.concatenate((x,[1]));
    p = sigmoid(f(w,x));
    if(p>0.5):
        return 1;
    return 0;


# ------------------------------------------------------------------------
#                               Batch Gradient Descent
# ------------------------------------------------------------------------
def SGD(X,Y,objective,derivative,iterations,lRate,bSize):
    # Initial Guess
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    m = len(X[0]);
    W = np.zeros(m);
    for i in range(iterations):
        # Print the function
        pred = [sigmoid(f(W,x)) for x in X];
        currLoss = objective(pred,Y);
#        print(currLoss);
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
                subgradientW[k] += derivative(pred[j],Y[j],X[j][k]);
        # Add the gradients to the weights    
        W -= lRate * subgradientW;
        # Lower the learning rate
        lRate = 0.99 * lRate;

    print("Gradient Descent optimization Complete\nIterations: ",i);    
    return W;

def Solve(X,Y,iterations=1000,lRate=0.1,bSize=5):
    return SGD(X,Y,logcost,derivative,iterations,lRate,bSize);