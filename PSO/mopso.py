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
    def __init__(self,dim,Bounds):
        self.pos = np.random.uniform(Bounds[0],Bounds[1],dim) 
        self.fitness = [0 for n in range(dim)]
        self.bestfitness = float("inf")
        self.bestpos = self.pos 
        self.vel = np.array([0,0])
        self.contribution = 0

def Fitness(x):
    return benchmarks.zdt1(x)

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

def RandomSelect(A,TB,alpha=0.20):
    if TB == "TOP":
        return rand.sample(A[:int(len(A)*alpha)],1)[0]
    if TB == "BOT":
        return rand.sample(A[int(len(A)*(1-alpha)):],1)[0]

def PSO(Swarm,MaxIters,w,c1,c2,Bounds,dim):
    
    print("\n\n\n\n\n\n")
    
    for particle in Swarm:
        particle.fitness = Fitness(particle.pos)
        
    A=Swarm.copy()

    for _ in trange(MaxIters): 
        
        #A = bentleynondomiatedsorting(A)
        A = fastnondomiatedsorting(A)
        refpoint = [ (max(A, key=lambda objectt: objectt.fitness[i]).fitness[i] + 0.1 ) for i in range(dim)]
        A = updatecontributions(A,refpoint)
        A.sort(key=lambda objectt: objectt.contribution,reverse = True)

        LastA = A.copy()

        for _ in range(len(Swarm)):

            globalbest = RandomSelect(A,'TOP')
            particle = RandomSelect(A,'BOT')

            for i in range(0):

                r1 = np.random.uniform(0.0000001,1)
                r2 = np.random.uniform(0.0000001,1)

                particle.vel[i] = w * particle.vel[i] + (c1 * r1 * (particle.bestpos[i] - particle.pos[i])) + (c2 * r2 * (globalbest.bestpos[i] - particle.pos[i]))
                particle.pos[i] = particle.pos[i] + particle.vel[i]

                if particle.pos[i][0] < Bounds[0] or particle.pos[i][0] > Bounds[1]:
                    particle.pos[i][0] = np.random.uniform(Bounds[0],Bounds[1])
                if particle.pos[i][1] < Bounds[2] or particle.pos[i][1] > Bounds[3]:
                    particle.pos[i][1] = np.random.uniform(Bounds[2],Bounds[3])
            
        for particle in Swarm:
            particle.fitness = Fitness(particle.pos)

        A = LastA + Swarm

    #A = bentleynondomiatedsorting(Swarm)
    A = fastnondomiatedsorting(A)
    refpoint = [ (max(A, key=lambda objectt: objectt.fitness[i]).fitness[i] + 0.1 ) for i in range(dim)]
    A = updatecontributions(A,refpoint)
    A.sort(key=lambda objectt: objectt.contribution, reverse = True)

    pareto = [ob.fitness for ob in A]
                
    bestsolever = min(Swarm, key=lambda objectt: objectt.bestfitness)
   
    return bestsolever.bestpos,bestsolever.bestfitness, pareto
        
def PSOClustering(nparticles = 1000, niterations = 200):

    w  = 0.001
    c1 = 0.0001
    c2 = 0.0001

    Bounds = [0, 1]

    print(Bounds)

    dim = 2

    Swarm = [particle(dim,Bounds) for i in range(nparticles)]

    solution, fitness, pareto = PSO(Swarm,niterations,w,c1,c2,Bounds,dim)

    return solution, fitness, pareto

#Main

bestpos, fitness, pareto = PSOClustering()
colors = np.random.rand(len(pareto))
x = np.linspace(0,1,1000)
y = 1 - sc.sqrt(x)
plt.plot(x,y,alpha=0.2)
plt.scatter([p[0] for p in pareto],[p[1] for p in pareto], s=5, c=colors, alpha=0.5) 
plt.show()