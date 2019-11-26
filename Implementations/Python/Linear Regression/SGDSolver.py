#%%
# Optimization - Minimization
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# Define the functions needed for "Shallow backpropogation" or simply gradient descent
# Regression loss, squared loss
def loss(y_hat,y):
  return 0.5 * (y_hat - y)**2;

# Regression Cost, Mean Squared Loss
def cost(Y_pred,Y):
  sum = 0;
  M = len(Y);
  for i in range(M):
    sum += loss(Y_pred[i], Y[i]);
  return sum/M;

def gradient(W,x,y,i):
  y_hat = np.inner(W,x);
  return (y_hat - y)*x[i];

def SGD(X,Y,Der,T,lRate,batchSize):
    # Add the psuedo-feature 1 for the bias calculation
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    m = len(X[0]);
    # Initialize the weights
    W = np.zeros((T,m));
    for t in range(T):
        # Batch mode (use 1 for incremental, use len(X) for deterministic gradient descent)
        batch = np.random.randint(-1,len(X),batchSize);
        for j in batch:
            grad = [Der(W[t-1],X[j],Y[j],i) for i in range(len(W[t-1]))];
            W[t] = W[t-1] - np.dot(lRate,grad);
    return np.mean(W,axis=0)

# Predict
def predict(W,X):
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    return np.inner(W,X);

#%%
# Create the mock data
X = np.array([[0.3,0.5],[1,1]]);
Y = np.array([-1,1]);
W = np.array([0,0,0]);

# Get the predictions
Y_pred = predict(W,X);
# Calculate the loss
print("Before optimization:",cost(Y_pred,Y))
W = SGD(X,Y,gradient,1000,0.5,1);
#%%
# Get the predictions
Y_pred = predict(W,X);
# Calculate the loss
print(cost(Y_pred,Y))