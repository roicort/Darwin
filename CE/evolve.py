import scipy as sc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def F( x, a=20, b=0.2, c=2*sc.pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( sc.cos( c * x ))
    return -a*sc.exp( -b*sc.sqrt( s1 / n )) - sc.exp( s2 / n ) + a + sc.exp(1)

def returnE(object):
    return object.E

class Individual:
    def __init__(self,n=2):
        self.X = np.array([random.randint(-30,30) for i in range(n)])
        self.E = F(self.X)
        self.S = np.random.normal(0, 1,n)

MaxGen=10000 #Number of Pseudosteps
n=20 #Dimention
P=2000 #Number of Individuals
epsilon=0.0001
alpha=2
fathers=[Individual(n) for i in range(P)]

t=0
while t < MaxGen:
    children=[]
    for father in fathers:

        newsigma = np.maximum(epsilon,father.S*(1+alpha*np.random.normal(0,1,n)))
        newx = father.X + newsigma*np.random.normal(0,1,n)

        child = Individual()
        child.X = newx
        child.S = newsigma
        child.E = F(child.X)

        children.append(child)

    population = fathers + children
    population.sort(key=returnE,reverse=False) #Order Firelfies
    fathers = population[:P]
    Bestyet = fathers[0]
    t+=1

print("")
print("X = "+str(Bestyet.X))
print("")
print("F(X) = "+str(Bestyet.E))
