# ------------------------------------------------------------
#                      Gradient Descent
# An iterative way to minimize a differentiable function
# W(t) = W(t-1) - n * f'(x)
# ------------------------------------------------------------

# Minimize f(x) = x^2
#%%
def objective(x):
    return x**2;
def derivative(x):
    return 2*x;
#%%
import matplotlib.pyplot as plt
import numpy as np
#%%
X = np.linspace(-1,1,1000);
plt.plot(X,objective(X))
#%%
figure = plt.figure();
xGuess = -0.7;
rate = 0.3;
for i in range(10):
    plt.plot(X,objective(X))
    plt.scatter(xGuess,objective(xGuess),color='red')
    plt.text(xGuess,objective(xGuess),str(i))
    xGuess -= rate*derivative(xGuess)
    print(rate)
    rate *= 0.95;

    