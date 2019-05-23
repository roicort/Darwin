import numpy as np 
import scipy as sc
import random as rand
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

rand.seed(1)
np.random.seed(1)

class particle:
    def __init__(self,points,ncentroids):
        self.pos = rand.sample(points,ncentroids) # position of all centroids
        self.times = [0 for i in range(ncentroids)]
        self.areas = [0 for i in range(ncentroids)]
        self.fitness = 0
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

    for _ in trange(MaxIters):  

        for particle in Swarm:
            particle.clusters = calcclusters(particle.pos,Points)
            #particle.areas, particle.radius = calcareas(particle.pos,particle.clusters,Points)
            particle.times = calctimes(particle.clusters,Points) 
            particle.fitness = Fitness(particle.times)
            #particle.fitness = Fitness(particle.areas) 
            if particle.fitness < particle.bestfitness:
                particle.bestfitness = particle.fitness
                particle.bestpos = particle.pos[:]

        globalbest=min(Swarm, key=lambda objectt: objectt.bestfitness)
        #plt.scatter([c[0] for c in globalbest.pos],[c[1] for c in globalbest.pos],s=5,c='red')
        #plt.show()
        tqdm.write("Fitness = %f" % globalbest.bestfitness)

        for particle in Swarm:
            for i in range(ncentroids):

                r1 = np.random.uniform(0.0000001,1)
                r2 = np.random.uniform(0.0000001,1)

                particle.vel[i] = w * particle.vel[i] + (c1 * r1 * (particle.bestpos[i] - particle.pos[i])) + (c2 * r2 * (globalbest.bestpos[i] - particle.pos[i]))
                particle.pos[i] = particle.pos[i] + particle.vel[i]

                if particle.pos[i][0] < Bounds[0] or particle.pos[i][0] > Bounds[1]:
                    particle.pos[i][0] = np.random.uniform(Bounds[0],Bounds[1])
                if particle.pos[i][1] < Bounds[2] or particle.pos[i][1] > Bounds[3]:
                    particle.pos[i][1] = np.random.uniform(Bounds[2],Bounds[3])

                
    bestsolever = min(Swarm, key=lambda objectt: objectt.bestfitness)
   
    return bestsolever.bestpos,bestsolever.bestfitness
        
def PSOClustering(Points, ncentroids = 13, nparticles = 25, niterations = 50):

    w  = 0.5
    c1 = 1
    c2 = 1
    
    centroids = []
    for p in range(len(Points)):
        centroids.append( np.array(Points[p][1:2][0]) ) 

    Swarm = [particle(centroids,ncentroids) for i in range(nparticles)]

    solution, fitness = PSO(Points,Swarm,niterations,w,c1,c2,ncentroids)

    return solution, fitness

"""
#Main

data = loaddata(plot=True)
centroids, fitness, radius = PSOClustering(data,niterations = 50)
#np.save(str(fitness),centroids)
plt.scatter([c[0] for c in centroids],[c[1] for c in centroids],s=5,c='red')
for i in range(len(centroids)):
    plt.Circle((centroids[i][0],centroids[i][1]),radius[i])
#plt.savefig(str(fitness)+".png")
plt.show()
"""

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

times = [[np.array([  20.54207775, -100.39886821]), np.array([  20.61614237, -100.38300083]), np.array([  20.55274171, -100.40801709]), np.array([  20.63883323, -100.454124  ]), np.array([  20.62749588, -100.40877338]), np.array([  20.59037274, -100.41587295]), np.array([  20.64583539, -100.40252767]), np.array([  20.64823524, -100.46735357]), np.array([  20.5718194 , -100.35987544]), np.array([  20.62422038, -100.45938508]), np.array([  20.5964187 , -100.40795505]), np.array([  20.62852259, -100.43364469]), np.array([  20.66918299, -100.46105845])], [np.array([  20.60249551, -100.43807808]), np.array([  20.59310002, -100.37078067]), np.array([  20.58076389, -100.3940339 ]), np.array([  20.64333026, -100.44629782]), np.array([  20.64884459, -100.39703913]), np.array([  20.62369428, -100.4266582 ]), np.array([  20.65421584, -100.47543626]), np.array([  20.64947728, -100.38617178]), np.array([  20.64944927, -100.46895431]), np.array([  20.61306068, -100.44970275]), np.array([  20.55063634, -100.38446619]), np.array([  20.55748031, -100.40684165]), np.array([  20.58079516, -100.39966683])], [np.array([  20.58207184, -100.41719624]), np.array([  20.64130137, -100.45823261]), np.array([  20.56958792, -100.39156275]), np.array([  20.6153612 , -100.39882639]), np.array([  20.56058516, -100.38569307]), np.array([  20.55856201, -100.39840824]), np.array([  20.62988489, -100.42445512]), np.array([  20.65521709, -100.36748136]), np.array([  20.65911982, -100.43552011]), np.array([  20.63704456, -100.45383405]), np.array([  20.61459737, -100.41327851]), np.array([  20.63095948, -100.46856944]), np.array([  20.60701498, -100.40196282])]]

areas = [[np.array([  20.65401504, -100.49744657]), np.array([  20.66180067, -100.42738314]), np.array([  20.60772655, -100.45245755]), np.array([  20.65000043, -100.35045002]), np.array([  20.72569768, -100.46172983]), np.array([  20.80169652, -100.44134032]), np.array([  20.61856741, -100.35837281]), np.array([  20.53540502, -100.40014165]), np.array([  20.78887142, -100.45606919]), np.array([  20.63691929, -100.4359274 ]), np.array([  20.5632744 , -100.38713976]), np.array([  20.57698727, -100.35968556]), np.array([  20.58715425, -100.37814134])], [np.array([  20.66685168, -100.47626952]), np.array([  20.60964227, -100.33523896]), np.array([  20.64287958, -100.34788043]), np.array([  20.69762941, -100.40186907]), np.array([  20.55386795, -100.39221826]), np.array([  20.5955225 , -100.41720502]), np.array([  20.57536428, -100.45204471]), np.array([  20.59161689, -100.44260568]), np.array([  20.74637579, -100.44572419]), np.array([  20.6327894 , -100.49671683]), np.array([  20.56259413, -100.33824419]), np.array([  20.62378914, -100.41174555]), np.array([  20.79124453, -100.45389453])], [np.array([  20.59691561, -100.3637336 ]), np.array([  20.68024792, -100.47328637]), np.array([  20.62570869, -100.37724059]), np.array([  20.6519213 , -100.33802151]), np.array([  20.5779386 , -100.36075888]), np.array([  20.66333151, -100.37222602]), np.array([  20.66619721, -100.47625319]), np.array([  20.62867627, -100.44473497]), np.array([  20.561895  , -100.42670628]), np.array([  20.79401695, -100.492803  ]), np.array([  20.77918101, -100.44319648]), np.array([  20.71442579, -100.47745694]), np.array([  20.55686421, -100.37575485])]]

Centroids0 = areas[0]
Centroids1 = areas[1]
Centroids2 = areas[2]

Bestfitnesstimes = [9337.517884142788, 3193.945270434756, 4159.309512563566]

Bestfitnessareas = [0.0009546803080499699, 0.00026132443860146104, 0.0005208312760768092]

data = loaddata(plot=True)

centroids = Centroids1

plt.scatter([c[0] for c in centroids],[c[1] for c in centroids],s=5,c='red')

plt.show()
