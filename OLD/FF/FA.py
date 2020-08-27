
#####################
# Firelfy Algorithm #
# By Rodrigo Cortez #
#####################

###############################################################################################

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy as sc
from scipy import linalg
from tests import *

###########################################################acl####################################

class Firefly:
    def __init__(self,n,Bounds):
        self.X = np.random.uniform(Bounds[0], Bounds[1], n)
        self.I = None

    def __repr__(self):
        return "Firelfy"+str(self.X)+" I"+" = "+str(self.I)

def distance(FireflyI,FireflyJ):
    R = linalg.norm(FireflyI - FireflyJ)
    return R

def calculateatractivness(FireflyI,FireflyJ,Gamma,Betazero):
    B = Betazero*sc.exp(-Gamma*distance(FireflyI.X,FireflyJ.X))
    return B

def move(FireflyI,FireflyJ,Beta,Gamma,Alpha,n,Bounds):
    X = (1.0 - Beta ) * FireflyI.X + Beta * FireflyJ.X
    for i in range(len(X)):
        if X[i] < Bounds[0]:
            X[i] = Bounds[0]
        elif X[i] > Bounds[1]:
            X[i] = Bounds[1]
    return X

###############################################################################################

def FA(Fireflies,MaxGen,Alpha,Gamma,n,Population,Bounds,Betazero):
    t=0
    while t < MaxGen:
        print("Gen "+str(t))
        for I in range(len(Fireflies)):
            for J in range(len(Fireflies)):
                if (Fireflies[I].I > Fireflies[J].I): 
                    Beta=calculateatractivness(Fireflies[I],Fireflies[J],Gamma,Betazero)
                    Fireflies[I].X = move(Fireflies[I],Fireflies[J],Beta,Gamma,Alpha,n,Bounds)
                    Fireflies[I].I = F(Fireflies[I].X)
            #End for j
        #End for  I
        t+=1
        Fireflies.sort(key=lambda objectt: objectt.I)
        Fireflies[0].X =  Fireflies[0].X + (Alpha * np.random.uniform(-1, 1, n))
        Fireflies[0].I = F(Fireflies[0].X)
    #End While  
    Fireflies.sort(key=lambda objectt: objectt.I) #Order Firelfies
    return Fireflies

###############################################################################################

def F(x):
    return rastrigin(x)

Bounds=rastriginbounds

"""
#Fast

Alpha = 0.1 #Randomness
Gamma = 0.1 #Light Absorption Coefficient
Betazero=1.0
MaxGen=100 #Number of Pseudosteps
n=2 #Dimention
Population=40 #Number of Fireflies
"""


#Accurate

Alpha = 0.01 #Randomness
Gamma = 0.1 #Light Absorption Coefficient
Betazero=1.0
MaxGen=1000 #Number of Pseudosteps
n=2 #Dimention
Population=40 #Number of Fireflies


"""
#+Vars

Alpha = 0.8 #Randomness
Gamma = 0.09 #Light Absorption Coefficient
Betazero=0.8
MaxGen=2000 #Number of Pseudosteps
n=10 #Dimention
Population=200 #Number of Fireflies
"""
###############################################################################################

Fireflies=[Firefly(n,Bounds) for i in range(Population)]

for f in range(len(Fireflies)):
    Fireflies[f].I = F(Fireflies[f].X)

InitialFireflies=deepcopy(Fireflies)

FinalFireflies=FA(Fireflies,MaxGen,Alpha,Gamma,n,Population,Bounds,Betazero)

BestFirefly = FinalFireflies[0] #BestFirefly

print("")
print("Best Firefly " + str(BestFirefly.X))
print("Best Result "+ str(F(BestFirefly.X)))
print("")

#for i in range(10):
#    print(FinalFireflies[i])

###############################################################################################

if n == 0:

    x1 = np.linspace(Bounds[0], Bounds[1])
    x2 = np.linspace(Bounds[0], Bounds[1])
    X1, X2 = np.meshgrid(x1,x2)
    F = F([X1,X2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1,X2,F, cmap=cm.rainbow,linewidth=0, antialiased=True)
    plt.suptitle('Ackley', fontsize=16)
    plt.grid(True)

    plt.show()

    plt.contour(X1,X2,F)
    plt.suptitle('Initial Positions', fontsize=16)
    plt.grid(True)

    for InitFire in InitialFireflies:
        plt.scatter(InitFire.X[0],InitFire.X[1], s=10)

    plt.show()

    plt.contour(X1,X2,F)
    plt.suptitle('Final Position', fontsize=16)
    plt.grid(True)

    for FinalFire in FinalFireflies:
        plt.scatter(FinalFire.X[0],FinalFire.X[1], s=10)

    #plt.scatter(BestFirefly.X[0],BestFirefly.X[1], s=10)

    plt.show()

###############################################################################################