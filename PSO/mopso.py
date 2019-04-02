import numpy as np 
import scipy as sc
import random as rand
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import pygmo as pg

def calchv(points):
    hv = hypervolume(points)
    ref_point = [2,2]
    return hv.compute(ref_point) , hv.contributions(ref_point) 

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

def PSO(Points,Swarm,MaxIters,w,c1,c2,ncentroids):
 
    xmin=min(Points, key=lambda list: list[1][0])[1][0]
    xmax=max(Points, key=lambda list: list[1][0])[1][0]
    ymin=min(Points, key=lambda list: list[1][1])[1][1]
    ymax=max(Points, key=lambda list: list[1][1])[1][1]

    Bounds = [xmin,xmax,ymin,ymax]

    print(Bounds)
    
    print("\n\n\n\n\n\n")

    A = [[] for _ in range(MaxIters)]

    for t in trange(MaxIters): 

        for particle in Swarm:

            particle.clusters = calcclusters(particle.pos,Points)
            particle.areas, particle.radius = calcareas(particle.pos,particle.clusters,Points)
            particle.times = calctimes(particle.clusters,Points) 

            particle.fitness[0] = Fitness(particle.times)
            particle.fitness[1] = Fitness(particle.areas) 

            if particle.fitness < particle.bestfitness:
                particle.bestfitness = particle.fitness
                particle.bestpos = particle.pos[:]

        HVC = HvContribution(At)
        At.order(decresaing)

        for particle in Swarm:
            globalbest = RandomSelect(At,TOP)
            pBest = RandomSelect(At,BOT)
            for i in range(ncentroids):

                r1 = np.random.uniform(0.0000001,1)
                r2 = np.random.uniform(0.0000001,1)

                particle.vel[i] = w * particle.vel[i] + (c1 * r1 * (particle.bestpos[i] - particle.pos[i])) + (c2 * r2 * (globalbest.bestpos[i] - particle.pos[i]))
                particle.pos[i] = particle.pos[i] + particle.vel[i]

                if particle.pos[i][0] < Bounds[0] or particle.pos[i][0] > Bounds[1]:
                    particle.pos[i][0] = np.random.uniform(Bounds[0],Bounds[1])
                if particle.pos[i][1] < Bounds[2] or particle.pos[i][1] > Bounds[3]:
                    particle.pos[i][1] = np.random.uniform(Bounds[2],Bounds[3])
                    
            particle.bound()
            particle.clusters = calcclusters(particle.pos,Points)
            particle.areas, particle.radius = calcareas(particle.pos,particle.clusters,Points)
            particle.times = calctimes(particle.clusters,Points) 

            particle.fitness[0] = Fitness(particle.times)
            particle.fitness[1] = Fitness(particle.areas) 

        for particle in Swarm:
            A[t+1] = UpdateExternalArchiveHv(At, xt+1)


                
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


"""
means=[]
deviations = []
sizes = [25,50,75]
bestfitnesses = []
bestcentroidss = []

for swarmsize in sizes:
    fintesses = []
    centroidss = []
    for _ in range(30):
        data = loaddata(plot=True)
        centroids, fitness = PSOClustering(data,nparticles=swarmsize)
        fintesses.append(fitness)
        centroidss.append(centroids)
        #np.save("solution"+str(swarmsize),centroids)
        #plt.scatter([c[0] for c in centroids],[c[1] for c in centroids],s=5,c='red')
        #plt.savefig("solution"+str(swarmsize)+".png")
    means.append(np.mean(fintesses))
    deviations.append(np.std(fintesses))
    bestfitnesses.append(min(fintesses))
    bestcentroidss.append(centroidss[np.argmin(np.array(fintesses))])

print("\n")
print(bestfitnesses)
print("\n")
print(means)
print("\n")
print(deviations)
print("\n")
print(bestcentroidss)
"""