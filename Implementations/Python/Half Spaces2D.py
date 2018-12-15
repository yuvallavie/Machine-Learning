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
import matplotlib.pyplot as plt
#%% Data Section
def f(x):
    if(x[0] + x[1] <= 1):
        return 1
    else:
        return -1

D = np.random.rand(500,2)
print("The Domain Space")
for x in D:
    if(f(x) == 1):
        plt.scatter(x[0],x[1],marker='^',color='b')
    else:
        plt.scatter(x[0],x[1],marker='o',color='r')
plt.show()

#%%
print("The Sample Space")
S = [D[i] for i in range(100)]
Y = [f(x) for x in S]
for i in range(len(S)):
    if(Y[i] == 1):
        plt.scatter(S[i][0],S[i][1],marker='^',color='b')
    else:
        plt.scatter(S[i][0],S[i][1],marker='o',color='r')
plt.show()
#%% Scikit Learn Linear Fit
from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(S, Y)
print("Prediction")
for x in D:
    if(clf.predict([[x[0],x[1]]]) == 1):
        plt.scatter(x[0],x[1],marker='^',color='b')
    else:
        plt.scatter(x[0],x[1],marker='o',color='r')
plt.show()
        
Risk = 0;
for i in range(len(D)):
    if(clf.predict([[D[i][0],D[i][1]]]) != f(D[i])):
        Risk = Risk + 1;
Risk = Risk/len(D)
    
    