import numpy as np

def Solve(x_train,y_train):
    # Save it as an NP Array
    X = np.array(x_train);
    # Add the ones to the end for the bias
    X = np.concatenate((X, np.ones((len(X),1))), axis=1);
    # Save labels as NP
    Y = np.array(y_train);
    
    N = len(X);
    # Set weights to all zeros
    W = np.zeros(len(X[0]));
    # since Perceptron convergeces we use a while loop
    flag = True;
    epochs = 0;
    updates = 0;
    while(flag):
        epochs += 1;
        flag = False;
        for i in range(N):
            ans = np.dot(W,X[i])*Y[i];
            if(ans <= 0):
                flag = True
                W += np.dot(Y[i], X[i])
                updates += 1;
    print(f"Perceptron terminated successfully after {epochs} epochs and {updates} updates with result W = {W}")
    return W;