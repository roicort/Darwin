
import scipy as sc
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
m=[]

for i in range(30):
    v=ee(n,k,c,MaxGen,sigma)
    m.append(F(v))

m.sort()
m.reverse()

median =sc.mean(m)
std = sc.std(m)
maxx=m[0]
minn=m[-1]

print("Mean = " +str(median))
print("Std = " +str(std))
print("Max Value = " + str(maxx))
print("Min Value = "+ str(minn))

Bounds = [-50,50]
x1 = np.linspace(Bounds[0], Bounds[1])
x2 = np.linspace(Bounds[0], Bounds[1])
X1, X2 = np.meshgrid(x1,x2)
F = F([X1,X2])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1,X2,F, cmap=cm.rainbow,linewidth=0, antialiased=True)
plt.grid(True)

plt.show()