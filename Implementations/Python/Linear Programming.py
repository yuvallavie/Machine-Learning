# --------------------------------------------
                # Linear Programming Solver
                # Scipy
# --------------------------------------------

#%%
# imports
from scipy.optimize import linprog    
import numpy as np
#%% 
# Minimize -X0 + 4X1
# Such That
# -3X0 + X1 <= 6
# -X0 - 2X1 >= -4
# X1 >= -3               
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
#%%
# Vectoried form
# Maximize <u,w>
# Such That
# Aw >= v
u = np.zeros(2);
v = np.ones(2);
A = np.ones((2,2));
for i in range(len(A)):
    for j in range(len(A[0])):
        A[i][j] = np.random.randn()

res2 = linprog(-u,A_ub=A,b_ub=v)