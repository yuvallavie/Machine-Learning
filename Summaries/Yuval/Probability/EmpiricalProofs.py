# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:22:07 2019

@author: Yuval Lavie
@title: Empirical Evidence for Theorems in Probability Theory
"""
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
#%%
print("Markov's Inequality")
print("---------------------------------------------------------------------")
print("Exponential Distribution PDF: L*e^(-Lx), CDF: 1 - e^(-Lx)), X > 0")
L = 1
X = np.random.exponential(L,10000)
print("P[X>=a] < E[X]/a")
a = 5
EX = np.mean(X)
prob = np.size([1 for x in X if x > a])/np.size(X)
print("P[X>=a] =",prob,"E[X]/a:",EX/a)



#%%