
import scipy as sc
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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

MaxGen=500 #Number of Pseudosteps
n=2 #Dimention
P=50 #Number of Individuals
epsilon=0.0001
alpha=2
me = []  
stats = []

for i in range(10):
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

    v=Bestyet
    me.append(v.E)
    res=(v.X, v.E)
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