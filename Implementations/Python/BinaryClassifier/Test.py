# ---------------------------------------------------------------------------
#               Test all implementations of binary classifier
# ---------------------------------------------------------------------------
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt

# Visualize the result
def getLineFromSGD(W,bias,X):
     return np.inner(-(W[0]/W[1]),X) - bias/W[1];
#%%
# Create mock separable realizable data

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
#%%
# ---------------------------------------------------------------------------
#                         Solve with Linear Programming
# ---------------------------------------------------------------------------
import LPSolver as lps;
Wlp = lps.Solve(X,Y)
#%%
# ---------------------------------------------------------------------------
#                         Solve with Perceptron
# ---------------------------------------------------------------------------
import PerceptronSolver as ps;
Wp = ps.Solve(X,Y)
#%%
# ---------------------------------------------------------------------------
#                         Solve with Gradient Descent
# ---------------------------------------------------------------------------
import SGDSolver as sgd;
Wsgd = sgd.Solve(X,Y)
#%%

# Visualize the results
fig = plt.figure(figsize = (10,10))
for i in range(len(X)):
    if(Y[i] == 1):
        plt.scatter(X[i][0],X[i][1],color='b');
    else:
        plt.scatter(X[i][0],X[i][1],color='r');
plt.title("Separable Case (Perceptron , LP), Non-Separable (SGD)")
# Prepare the linear space
ls = np.linspace(np.min(X),np.max(X));

# Plot the Linear Programming solution
plt.plot(ls,getLineFromSGD(Wlp,Wlp[2],ls),label='Linear Programming')

# Plot the Perceptron solution
plt.plot(ls,getLineFromSGD(Wp,Wp[2],ls),label='Perceptron')

# Plot the Stochastic Gradient Descent solution
plt.plot(ls,getLineFromSGD(Wsgd[0],Wsgd[1],ls),label='SGD')

plt.legend(loc='upper left')
plt.xlabel("X1")
plt.ylabel("X2");
plt.show()