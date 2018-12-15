# ----------------------------------------------------------
                # Basic Linear Classifier
                # H = sign(<w,x> + b) - Half Spaces
                # L_D(h) - Px~D[h(x) != y]
                # L_S(h) - 0/1 Loss
                # Surrogate Loss - Hinge Loss = max(0,1-(f(x_i)*y)
                # f(x_i) = {-1,1}
                # The Realizeable Case
                # m(e,d) = 
# ----------------------------------------------------------
#%% Imports
import numpy as np                
#%% Data Section
n = 5 # Dimension
ds = 10**4 # Domain Size - (Unknown in real situations)
eps = 0.1 # Requested accuracy of the learner
delta = 2/100 # Probability the learner will fail
vcDim = n+1;
m = int((vcDim + np.log(1/delta))/eps) # Sample Size

def f(x):
    if(np.sum(x) < 1):
        return 1
    else:
        return -1

D = np.random.rand(ds,n)
#%% The Sample Space

S = [D[i] for i in range(m)]
Y = [f(x) for x in S]
#%% Scikit Learn Linear Fit
from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(S, Y)
#%%
print("Prediction")      
Risk = 0;
for i in range(len(D)):
    vector = [D[i][j] for j in range(n)]
    if(clf.predict([vector]) != f(D[i])):
        Risk = Risk + 1;
Risk = Risk/len(D)
    
    