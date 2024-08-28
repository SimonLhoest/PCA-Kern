# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:59:09 2022

@author: lhoes
"""

import numpy as np
import matplotlib.pyplot as plt
#%% Noyau polynominal
def kernp(x,y,c=7.5, d=2):
    return (c+x.T@y)**d
#%% Noyau gaussien
def kerng(x,y,c=1):
    return np.exp(-(1/(2*c**2))*np.linalg.norm(x - y)**2)
#%% Noyau sigmoide
def kerns(x,y,c=0,gamma=0.2):
    return np.tanh(gamma + c*x.T@y)
#%% Noyau exponentiel
def kerne(x,y):
    return np.exp(x.T@y)