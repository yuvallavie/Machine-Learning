# ---------------------------------------------------------------
#                      Logistic Regression
# ---------------------------------------------------------------
#%%
# imports

from LogisticRegression import Solve,predict;
import numpy as np;
import matplotlib.pyplot as plt
def LinearSeparator(W,bias,X):
     return np.inner(-(W[0]/W[1]),X) - bias/W[1];
#%%
#%%

# Simple mock data

M = 100;
X1 = 0.8*np.random.randn(int(M/2),2) + 2;
X2 = 0.8*np.random.randn(int(M/2),2) - 2;
X = np.concatenate((X1,X2));

# Create the labels
Y = np.zeros(M);
for i in range(M):
    if(X[i][0] + X[i][1] > 0.5):
        Y[i] = 1;
    else:
        Y[i] = -1;

for i in range(len(X)):
    if(Y[i] == 1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.show()

#%%

#%%
W = Solve(X,Y);
#%%
# Get predictions
yHat = [predict(W,x) for x in X]

#%%
# Visualize the results

# Visualize the results
fig = plt.figure(figsize = (10,10))
for i in range(len(X)):
    if(Y[i] == 1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.title("Logistic Regression")
# Prepare the linear space
ls = np.linspace(np.min(X),np.max(X));

# Plot the Logistic Regression solution
plt.plot(ls,LinearSeparator(W,W[2],ls),label='Logistic Regression')