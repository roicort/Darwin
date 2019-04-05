import numpy as np 
import scipy as sc
import random as rand
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import pygmo as pg
from pygmo import hypervolume

class particle:
    def __init__(self,points,ncentroids):
        self.pos = rand.sample(points,ncentroids) # position of all centroids
        self.times = [0 for i in range(ncentroids)]
        self.areas = [0 for i in range(ncentroids)]
        self.fitness = [0,0]
        self.bestfitness = float("inf")
        self.bestpos = self.pos #best pos of centroids
        self.vel = [np.array([0,0]) for i in range(ncentroids)]
        self.clusters = None
        self.radius = [0 for i in range(ncentroids)]
        self.contribution = 0

def loaddata(datapath="data.csv",plot=False):
    data = pd.read_csv(datapath) 
    df = data[['pdv_code','lat','lon','demanda','frecuencia']]
    df['tpsem'] = df['demanda'] * df['frecuencia']
    df.drop('demanda', axis=1, inplace=True)
    df.drop('frecuencia', axis=1, inplace=True)
    if plot:
        plt.scatter(df['lat'],df['lon'],s=0.50)
    points = df.values.tolist()
    for p in range(len(points)):
        points[p] = [points[p][0] , np.array(points[p][1:3]), points[p][3]]
    return points
    
def Fitness(algo):
    return np.std(algo)

def calctimes(clusters,points):
    times = []  
    for cluster in clusters:
        auxsum=0
        for index in cluster:
            auxsum+=points[index][2]
        times.append(auxsum)
    return times

def calcareas(centroids,clusters,points):
    areas = []  
    rads = []
    for centroid in range(len(centroids)):
        ra = -1
        for index in clusters[centroid]:
            dist = np.linalg.norm(centroids[centroid] - points[index][1])
            if dist > ra:
                ra = dist
        area = sc.pi*pow(2,ra)
        areas.append(area)
        rads.append(ra)
    return areas,rads

def calcclusters(centroids,points):
    clusters=[[] for i in range(len(centroids))]
    for po in range(len(points)):
        distmin = linalg.norm(points[po][1] - centroids[0])
        kmin = 0
        for c in range(1,len(centroids)):
            dist = linalg.norm(points[po][1] - centroids[c])
            if dist < distmin:
                kmin = c
                distmin = dist
        clusters[kmin].append(po)
    return clusters

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
                aux = nondominated[pn].copy()
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

def updatecontributions(A,refpoint):
    points = [ob.fitness for ob in A]
    hv = hypervolume(points)
    cont = hv.contributions(refpoint) 
    for o in range(len(A)):
        A[o].contribution = cont[o]
    return A

def RandomSelect(A,TB):
    if TB == "TOP":
        return rand.sample(A[0:len(A)/2],1)
    if TB == "BOT":
        return rand.sample(A[len(A)/2:],1)

def PSO(Points,Swarm,MaxIters,w,c1,c2,ncentroids):
 
    xmin=min(Points, key=lambda list: list[1][0])[1][0]
    xmax=max(Points, key=lambda list: list[1][0])[1][0]
    ymin=min(Points, key=lambda list: list[1][1])[1][1]
    ymax=max(Points, key=lambda list: list[1][1])[1][1]

    Bounds = [xmin,xmax,ymin,ymax]

    refpoint = [xmax,ymax]

    print(Bounds)
    
    print("\n\n\n\n\n\n")

    for t in trange(MaxIters): 

        for particle in Swarm:
            particle.clusters = calcclusters(particle.pos,Points)
            particle.areas, particle.radius = calcareas(particle.pos,particle.clusters,Points)
            particle.times = calctimes(particle.clusters,Points) 

            particle.fitness[0] = Fitness(particle.times)
            particle.fitness[1] = Fitness(particle.areas) 
        

        A = bentleynondomiatedsorting(Swarm)
        A = updatecontributions(A,refpoint)
        A.sort(key=lambda objectt: objectt.contribution)

        LastA = A.copy()

        for particle in Swarm:
            globalbest = RandomSelect(A,'TOP')
            particle.bestpos = RandomSelect(A,'BOT')

            for i in range(ncentroids):

                r1 = np.random.uniform(0.0000001,1)
                r2 = np.random.uniform(0.0000001,1)

                particle.vel[i] = w * particle.vel[i] + (c1 * r1 * (particle.bestpos[i] - particle.pos[i])) + (c2 * r2 * (globalbest.bestpos[i] - particle.pos[i]))
                particle.pos[i] = particle.pos[i] + particle.vel[i]

                if particle.pos[i][0] < Bounds[0] or particle.pos[i][0] > Bounds[1]:
                    particle.pos[i][0] = np.random.uniform(Bounds[0],Bounds[1])
                if particle.pos[i][1] < Bounds[2] or particle.pos[i][1] > Bounds[3]:
                    particle.pos[i][1] = np.random.uniform(Bounds[2],Bounds[3])

        for particle in Swarm:
            A = LastA + A
                
    bestsolever = min(Swarm, key=lambda objectt: objectt.bestfitness)
   
    return bestsolever.bestpos,bestsolever.bestfitness,bestsolever.radius
        
def PSOClustering(Points, ncentroids = 13, nparticles = 25, niterations = 50):

    w  = 0.5
    c1 = 1
    c2 = 1

    centroids = []
    for p in range(len(Points)):
        centroids.append( np.array(Points[p][1:2][0]) ) 

    Swarm = [particle(centroids,ncentroids) for i in range(nparticles)]

    solution, fitness, radius = PSO(Points,Swarm,niterations,w,c1,c2,ncentroids)

    return solution, fitness, radius

#Main

data = loaddata(plot=True)
centroids, fitness, radius = PSOClustering(data,niterations = 1)
#np.save(str(fitness),centroids)
plt.scatter([c[0] for c in centroids],[c[1] for c in centroids],s=5,c='red')
for i in range(len(centroids)):
    plt.Circle((centroids[i][0],centroids[i][1]),radius[i])
#plt.savefig(str(fitness)+".png")
plt.show()