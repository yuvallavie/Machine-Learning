# -----------------------------------------------------------------------------
#                   Binary Classifer with a neural network mindset
# -----------------------------------------------------------------------------
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import Helpers
import StochasticGradientDescent as sgd
#%%
# Create mock data
X,Y = Helpers.CreateLinearSeparableMockData(2,100,0.03);
Helpers.VisualizeData(X,Y)
#%%
[W,b] = sgd.HingeLoss(X,Y,maxIter=1000,lRate = 1);
ls = np.linspace(np.min(X),np.max(X));
sgd.VisualizeResult2D(W,b,X,Y)
#%%
def f(W,x,b):
    return np.inner(W,x) + b;
def sigmoid(x):
    return (1 / (1+np.exp(-x)));
#%%
ls2 = np.linspace(-5,5);
plt.plot(ls2,sigmoid(ls2))
plt.grid(True)
plt.show()
#%%
def LogisticLoss(yHat,y):
    return -y*np.log(yHat) - (1-y)*np.log(1 - yHat);

def LogisticDerivativeW()