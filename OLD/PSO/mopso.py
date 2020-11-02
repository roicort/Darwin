import numpy as np 
import scipy as sc
import random as rand
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from copy import deepcopy

import pygmo as pg
from pygmo import hypervolume
from deap import benchmarks

class particle:
    def __init__(self,dim,n_funct,Bounds):
        self.pos = np.random.uniform(Bounds[0],Bounds[1],dim) 
        self.fitness = [0 for n in range(n_funct)]
        self.vel = np.array([0 for _ in range(dim)])
        self.contribution = 0

def Fitness(x):
    return benchmarks.fonseca(x)

def dominance(A,B):

    contA = 0
    contB = 0

    for k in range(len(A)):
        if A[k] < B[k]:
            contA+=1
        else:
            if B[k] < A[k]:
                contB+=1
    
    if contA > 0 and contB == 0:
        return 1
    if contB > 0 and contA == 0:
        return -1
    if contA == 0 and contB == 0:
        return 2
    return 0

def bentleynondomiatedsorting(population):
    nondominated = [population[0]]
    for p in range(len(population)):
        popaux = True
        pn=0
        while pn < len(nondominated):
            aux=None
            if dominance(nondominated[pn].fitness,population[p].fitness) == 1:
                aux = deepcopy(nondominated[pn])
                nondominated.pop(pn)
                nondominated.insert(0,aux)
                popaux = False
                break
            if dominance(population[p].fitness,nondominated[pn].fitness) == 1:
                nondominated.pop(pn)
            else:
                pn+=1
        if popaux:
            nondominated.append(population[p])
    return nondominated[1:]

def fastnondomiatedsorting(Swarm):
    points = [s.fitness for s in Swarm]
    ndf, _, _, _ = pg.fast_non_dominated_sorting(points)
    return [Swarm[i] for i in ndf[0]]

def updatecontributions(A,refpoint):
    points = [ob.fitness for ob in A]
    hv = hypervolume(points)
    cont = hv.contributions(refpoint) 
    for o in range(len(A)):
        A[o].contribution = cont[o]
    return A

def RandomSelect(A,TB,alpha=0.80):

    if TB == "TOP":
        return rand.sample(A[:int(len(A)*alpha)],1)[0]
    if TB == "BOT":
        return rand.sample(A[int(len(A)*(1-alpha)):],1)[0]

def getindex(particle,A):
    index = -1
    for i in range(len(A)):
        if A[i] == particle:
            index = i
    return index

def selectneighbor(A,particle):
    index = getindex(particle,A)
    if index == 0:
        return A[index+1]
    if index == len(A):
        return A[index-1]
    else:
        return A[index+np.random.randint(-1, 1)]


def PSO(Swarm,MaxIters,w,c1,c2,Bounds,dim,n_funct):
    
    print("\n\n\n\n\n\n")
    
    for particle in Swarm:
        particle.fitness = Fitness(particle.pos)
        
    A=Swarm.copy()

    for _ in trange(MaxIters): 
        
        #A = bentleynondomiatedsorting(A)
        A = fastnondomiatedsorting(A)
        refpoint = [ (max(A, key=lambda objectt: objectt.fitness[i]).fitness[i] + 0.1 ) for i in range(n_funct)]
        A = updatecontributions(A,refpoint)
        A.sort(key=lambda objectt: objectt.contribution,reverse = True)
        if len(A) > len(Swarm):
            A = A[:len(Swarm)]
        
        LastA = A.copy()

        for particle in Swarm:

            globalbest = RandomSelect(A,'TOP')
            localparticle = RandomSelect(A,'BOT')

            r1 = np.random.uniform(0.0000001,1)
            r2 = np.random.uniform(0.0000001,1)

            particle.vel = w * particle.vel + (c1 * r1 * (localparticle.pos - particle.pos)) + (c2 * r2 * (globalbest.pos - particle.pos))
            #particle.vel = w * particle.vel + (c2 * r2 * (globalbest.pos - particle.pos))
            particle.pos = particle.pos + particle.vel

            for i in range(len(particle.pos)):
                if particle.pos[i] < Bounds[0]:
                    particle.pos[i] = Bounds[0]
                if particle.pos[i] > Bounds[1]:
                    particle.pos[i] = Bounds[1]
            
            particle.fitness = Fitness(particle.pos)

        A = LastA + Swarm

    #A = bentleynondomiatedsorting(Swarm)
    A = fastnondomiatedsorting(A)
    refpoint = [ (max(A, key=lambda objectt: objectt.fitness[i]).fitness[i] + 0.1 ) for i in range(n_funct)]
    #refpoint = [1.5,1.5]
    A = updatecontributions(A,refpoint)
    A.sort(key=lambda objectt: objectt.contribution, reverse = True)
    if len(A) > len(Swarm):
         A = A[:100]

    pareto = [ob.fitness for ob in A]

    hv = hypervolume(pareto)
    hv = hv.compute(refpoint)
             
    bestsolever = min(Swarm, key=lambda objectt: objectt.fitness)

    return bestsolever.pos,bestsolever.fitness, pareto, hv 
        
def MOPSO(nparticles = 300, niterations = 200):

    w  = 0.4
    c1 = 1
    c2 = 1

    Bounds = [0, 1]

    dim = 30
    n_funct = 2

    Swarm = [particle(dim,n_funct,Bounds) for i in range(nparticles)]

    solution, fitness, pareto, hv = PSO(Swarm,niterations,w,c1,c2,Bounds,dim,n_funct)
    return solution, fitness, pareto, hv

#Main

hvs = []
bestpos, fitness, pareto, hv = MOPSO()
hvs.append(hv)
#print("")
print("Hypervolume = "+str(hv))
colors = np.random.rand(len(pareto))
x = np.linspace(0,1,1000)
plt.scatter([p[0] for p in pareto],[p[1] for p in pareto], s=5, c=colors, alpha=0.5) 
plt.show()

""" print("Mean " +str(np.mean(hvs)))
print("Std " +str(np.std(hvs))) """