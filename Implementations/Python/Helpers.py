# ------------------------------------------------------------
#                   Helper Functions for ML
# ------------------------------------------------------------

#%%
# imports
import numpy as np
import matplotlib.pyplot as plt;
#%%

# ------------------------------------------------------------
#                   Separable Data
# ------------------------------------------------------------
def CreateLinearSeparableMockData(dimension,batchSize,treshold):
    X1 =  np.random.randn(batchSize,dimension)*0.05  + 2*treshold;
    X2 =  np.random.randn(batchSize,dimension)*0.05  - 2*treshold;
    X = np.concatenate((X1,X2));
    Y = np.zeros(2*batchSize);
    for i in range(len(X)):
        if(np.sum(X[i]) >= treshold):
            Y[i] = 1;
        else:
            Y[i] = -1;
    return [X,Y];



# ------------------------------------------------------------
#                   Visualize Data
# ------------------------------------------------------------
def VisualizeData(X,Y):
    for i in range(len(X)):
        if(Y[i] == -1):
            plt.scatter(X[i][0],X[i][1],color='r');
        else:
            plt.scatter(X[i][0],X[i][1],color='b');
    plt.show();
#%%

