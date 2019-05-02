
#####################
# Firelfy Algorithm #
# By Rodrigo Cortez #
#####################

###############################################################################################
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy import linalg
import pygmo as pg
import random as r
from deap import benchmarks
from tests import *

###############################################################################################

class Firefly:
    def __init__(self,n,Bounds):
        self.X = np.random.uniform(Bounds[0], Bounds[1], n)
        self.I = None
        self.P = None

    def __repr__(self):
        return "DV"+str(self.X)+" Fitness"+" = "+str(self.I)

def distance(FireflyI,FireflyJ):
    R = linalg.norm(FireflyI - FireflyJ)
    return R

def calculateatractivness(FireflyI,FireflyJ,Gamma,Betazero):
    B = Betazero*sc.exp(-Gamma*pow(distance(FireflyI.X,FireflyJ.X),2))
    return B

def move(FireflyI,FireflyJ,Beta,Gamma,Alpha,n,Bounds):
    X = FireflyI.X + Beta * (FireflyJ.X-FireflyI.X)
    for i in range(len(X)):
        if X[i] < Bounds[0] or X[i] > Bounds[1]:
            X = np.random.uniform(Bounds[0], Bounds[1], n)
    return X

def dominance(FireflyJ,FireflyI):
    return pg.pareto_dominance(obj1 = FireflyJ.I, obj2 = FireflyI.I)
    #Returns true if obj1 dominates obj2

def genw(n):
    inte = 1
    a = []
    for _ in range(n-1):
        a += [r.random()*inte]
        inte  -= a[-1]
    a += [inte]
    r.shuffle(a)
    return a

def psi(w,f):
    sm=0
    for k in range(len(w)):
        sm+=w[k]*f[k]
    return sm

def indexmin(Fireflies):
    minv = 10**9+7
    index = -1
    for i in range(len(Fireflies)):
        if minv > Fireflies[i].P:
            index = i
            minv = Fireflies[i].P
    return index

def genpopulation(N,Bounds,P):
    Population = [Firefly(N,Bounds) for i in range(P)]
    for f in range(len(Population)):
        Population[f].I = F(Population[f].X)
    return Population

###############################################################################################

def MOFA(Fireflies,MaxGen,InitAlpha,Gamma,n,Population,Bounds,Betazero,Numfunc):
    start_time = time.time()
    t=0
    while t < MaxGen:
        print("Gen "+str(t))
        Alpha=InitAlpha*pow(0.9,t)
        MovingFire=deepcopy(Fireflies)
        for I in range(Population):
            Nondominated = False
            for J in range(Population):
                if dominance(Fireflies[J],Fireflies[I]) and I!=J:
                    Beta=calculateatractivness(Fireflies[I],Fireflies[J],Gamma,Betazero)
                    MovingFire[I].X = move(Fireflies[I],Fireflies[J],Beta,Gamma,Alpha,n,Bounds)
                    MovingFire[I].I = F(Fireflies[I].X)
                    Nondominated = True
            if not Nondominated:
                W = genw(Numfunc)
                for w in range(len(Fireflies)):
                    Fireflies[w].P = psi(W,Fireflies[w].I)
                Best = indexmin(Fireflies)
                MovingFire[Best].X =  Fireflies[Best].X + (Alpha * np.random.uniform(-1, 1, n))
                for i in range(len(Fireflies[Best].X)):
                    if MovingFire[Best].X[i] < Bounds[0]:
                        MovingFire[Best].X[i] = Bounds[0]
                    elif MovingFire[Best].X[i] > Bounds[1]:
                        MovingFire[Best].X[i] = Bounds[1]
                MovingFire[Best].I = F(MovingFire[Best].X)
            #End for j
        #End for  I
        Fireflies=deepcopy(MovingFire)
        Sort = list(pg.sort_population_mo([Fire.I for Fire in Fireflies]))
        Fireflies = [Fireflies[Sort[i]] for i in range(len(Sort))]
        #pg.plot_non_dominated_fronts([Fire.I for Fire in Fireflies])
        t+=1
    #End While

    hv = pg.hypervolume([Fire.I for Fire in Fireflies])
    ref_point = [11,11]
    hvv = hv.compute(ref_point)
    print("\nDone in %f seconds" % (time.time() - start_time) + "\n")

    return Fireflies, hvv

###############################################################################################

def F(x):
    return benchmarks.zdt2(x)

Bounds=[0.0,1.0]
Alpha = 0.1 #Randomness
Gamma = 1.0 #Light Absorption Coefficient
Betazero=0.7
MaxGen=1000 #Number of Pseudosteps
N=30 #Dimention
Numfunc = 2
P=100 #Number of Fireflies

Fireflies=genpopulation(N,Bounds,P)

Original=deepcopy(Fireflies)

Evolved, HV = MOFA(Fireflies,MaxGen,Alpha,Gamma,N,P,Bounds,Betazero,Numfunc)

print("Hypervolume "+str(HV)+"\n")

pg.plot_non_dominated_fronts([Fire.I for Fire in Original])
pg.plot_non_dominated_fronts([Fire.I for Fire in Evolved])

###############################################################################################