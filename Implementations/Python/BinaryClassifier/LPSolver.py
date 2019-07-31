import numpy as np
from scipy.optimize import linprog

# Returns the weights including the bias as the last term (w(0),w(1),...,w(n),b)
def Solve(X,Y):
        
    # Add the bias to the weights
    x_train = np.concatenate((X, np.ones((len(X),1))), axis=1);
    
    # Create the linear program
    # Max <u,w> s.t Aw>v
    cols = len(x_train[0]);    
    u = np.zeros(cols)
    rows = len(x_train)
    v = np.ones(rows)
    # Create the matrix
    A = np.zeros((rows,cols))
    for i in range(rows):
        A[i] = np.inner(x_train[i],Y[i]);
        
    # Create the bounds
    bounds = [];
    for i in range(len(A[0])):
        bounds.append((None,None));
    res = linprog(-u, A_ub = -A, b_ub = -v,bounds=bounds, options={"disp": True},method='simplex')
    return res.x;