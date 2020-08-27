
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import scipy as sc

def wood(X):

    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]

    summ = sum((
        100*(x1*x1 - x2)**2,
        (x1-1)**2,
        (x3-1)**2,
        90*(x3*x3 - x4)**2,
        10.1*((x2-1)**2 + (x4-1)**2),
        19.8*(x2-1)*(x4-1),
        ))
    
    return summ