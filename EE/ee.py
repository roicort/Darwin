
import scipy as sc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def F(X):
    return ((1.5 - X[0]*(1-X[1]))**2 ) + ((2.25 - X[0]*(1-X[1]**2))**2) +((2.625 - X[0]*(1-X[1]**3))**2)

def mutate(x,sigma,n):
    return x + sigma * np.random.normal(0, 1, n)

def ee(n,k,c,MaxGen,sigma):
    t = 0
    msuccessful = 0
    x = np.random.normal(0, 4.5, n)
    while t <= MaxGen:
        newx = mutate(x,sigma,n)
        if F(newx) < F(x):
            x = newx
            msuccessful+=1
        t+=1
        if t % k == 0:
            percent = msuccessful/k
            if percent > 1/5:
                sigma=sigma/c
            if percent < 1/5:
                sigma=sigma*c
            msuccessful=0
    return x

n=2 #Dimention
k=10
c=0.817
MaxGen=500
sigma=3.0

stats=[]
me=[]

for i in range(30):
    v=ee(n,k,c,MaxGen,sigma)
    me.append(v)
    res=(v,F(v))
    stats.append(res)

m=sorted(stats, key=lambda tup: tup[1])
m.reverse()

median =sc.mean(res)
std = sc.std(res)
maxx=m[0]
minn=m[-1]

print("Mean = " +str(median))
print("Std = " +str(std))
print("Max Value = " + str(maxx))
print("Min Value = "+ str(minn))
