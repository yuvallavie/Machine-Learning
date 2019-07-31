# --------------------------------------------------------
#                   Half Space Classification
#                   from scratch
# --------------------------------------------------------

#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
#X = [[0.3,0.3],[0.2,0.1],[0.3,0.4],[0.4,0.9]];
M = 10**1;
X = np.random.randn(M,2)
Y = np.zeros(M);
for i in range(M):
    if(X[i][0] + X[i][1] > 0.5):
        Y[i] = 1;
    else:
        Y[i] = -1;

for i in range(len(X)):
    if(Y[i] == -1):
        plt.scatter(X[i][0],X[i][1] ,Color='r')
    else:
        plt.scatter(X[i][0],X[i][1] ,Color='b')
plt.show()

#%%
def linearSeparator(weights,X,bias):
    return np.dot(weights,X) + bias;

def halfSpace(weights,X,b):
    return np.sign(linearSeparator(weights,X,b));

def Loss01(yHat,y):
    if(y==yHat):
        return 0;
    return 1;
#%%
for i in range(len(X)):
    if(Y[i] == -1):
        plt.scatter(X[i][0],X[i][1] ,Color='r')
    else:
        plt.scatter(X[i][0],X[i][1] ,Color='b')
#%%
# Perceptron
def Perceptron(x_train,y_train):
    # Save it as an NP Array
    X = np.array(x_train);
    # Add the ones to the end for the bias
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    # Save labels as NP
    Y = np.array(y_train);
    
    N = len(X);
    # Set weights to all zeros
    W = np.zeros(len(X[0]));
    flag = True;
    #for t in range(10):
    while(flag):
        flag = False;
        for i in range(N):
            print("W : " + str(W));
            print("X(i) = " + str(X[i]))
            ans = np.dot(W,X[i])*Y[i];
            print("y*<W,X(i) = " + str(ans));
            if(ans <= 0):
                flag = True
                W += np.dot(Y[i], X[i])
    return W[0:-1],W[-1];
#%%
W,b = Perceptron(X,Y)
yHat = [halfSpace(W,X[i],b) for i in range(len(X))]

        
#%%
