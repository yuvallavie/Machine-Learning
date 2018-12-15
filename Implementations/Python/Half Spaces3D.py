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
from mpl_toolkits import mplot3d
#%% Data Section
def f(x):
    if(x[0] + x[1] + x[2] < 1):
        return 1
    else:
        return -1

D = np.random.rand(300,3)
print("The Domain Space")

axD = plt.axes(projection='3d')

for x in D:
    if(f(x) == 1):
        axD.scatter3D(x[0],x[1],x[2],marker='^',color='b')
    else:
        axD.scatter3D(x[0],x[1],x[2],marker='o',color='r')
plt.show()

#%%
print("The Sample Space")
S = [D[i] for i in range(100)]
Y = [f(x) for x in S]
axS = plt.axes(projection='3d')
for i in range(len(S)):
    if(Y[i] == 1):
        axS.scatter3D(S[i][0],S[i][1],S[i][2],marker='^',color='b')
    else:
        axS.scatter3D(S[i][0],S[i][1],S[i][2],marker='o',color='r')
plt.show()
#%% Scikit Learn Linear Fit
from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(S, Y)
print("Prediction")
axP = plt.axes(projection='3d')
for x in D:
    if(clf.predict([[x[0],x[1],x[2]]]) == 1):
        axP.scatter3D(x[0],x[1],x[2],marker='^',color='b')
    else:
        axP.scatter3D(x[0],x[1],x[2],marker='o',color='r')
plt.show()
        
Risk = 0;
for i in range(len(D)):
    if(clf.predict([[D[i][0],D[i][1],D[i][2]]]) != f(D[i])):
        Risk = Risk + 1;
Risk = Risk/len(D)
    
    