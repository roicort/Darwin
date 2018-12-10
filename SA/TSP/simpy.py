"""

Simulated Annealing (STP)

Developped by Rodrigo Cortez
ENES Morelia, 2018

"""

import scipy as sc
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def plotcities(cities):
    data = {"x":[], "y":[], "label":[]}
    for city in cities:
        label=city[0]
        coord=city[1]
        data["x"].append(coord[0])
        data["y"].append(coord[1])
        data["label"].append(label)

    # display scatter plot data
    plt.figure(figsize=(10,8))
    plt.title('Map', fontsize=20)
    plt.scatter(data["x"], data["y"], marker = 'o')

    # add labels
    for label, x, y in zip(data["label"], data["x"], data["y"]):
        plt.annotate(label, xy = (x, y))

def plotsol(paths):
    for sol in paths:
        plotcities(cities)
        for city in range(len(sol)-1):
            plt.plot([sol[city][0],sol[city+1][0]], [sol[city][1],sol[city+1][1]])
        plt.plot([sol[-1][0],sol[0][0]], [sol[-1][1],sol[0][1]])
        plt.show()

def chooseneighboor(sol):
    a=random.randint(1,len(sol)-1)
    b=random.randint(1,len(sol)-1)
    sigrid=sol.copy()
    sigrid[a],sigrid[b] = sigrid[b],sigrid[a]
    return sigrid

def distance(c1, c2):
    dist = linalg.norm(np.array(c1) - np.array(c2))
    return dist

def calctemp(T):
    return (T*0.1)

def f(sol):
    return sum([distance(sol[i],sol[i+1]) for i in range(len(sol)-1)]) + distance(sol[-1],sol[0])

def initsol(cities):
    sol=[city[1] for city in cities]
    return sol

def simulatedannealing(T,mintemp,maxepoch,sol):
    t=0
    X=[sol]
    while T > mintemp and t < maxepoch:
        print("Epoch " +  str(t) +" "+ "Temp "+str(T))
        j=chooseneighboor(sol) #elegir j
        if f(j) <= f(X[-1]):
            X.append(j)

        if random.randint(0,1) < sc.exp((-f(j)-f(X[-1])/T)):
            X.append(j)
        t+=1
        T = calctemp(T)
    return X

#Main

# List latitude and longitude (degrees) for the twenty largest U.S. cities
cities = [['New York City',(40.72,74.00)], ['Los Angeles',(34.05,118.25)],
['Chicago', (41.88,87.63)], ['Houston', (29.77,95.38)],
['Phoenix', (33.45,112.07)], ['Philadelphia', (39.95,75.17)],
['San Antonio', (29.53,98.47)], ['Dallas', (32.78,96.80)],
['San Diego', (32.78,117.15)], ['San Jose', (37.30,121.87)],
['Detroit', (42.33,83.05)], ['San Francisco', (37.78,122.42)],
['Jacksonville', (30.32,81.70)], ['Indianapolis', (39.78,86.15)],
['Austin', (30.27,97.77)], ['Columbus', (39.98,82.98)],
['Fort Worth', (32.75,97.33)], ['Charlotte', (35.23,80.85)],
['Memphis', (35.12,89.97)], ['Baltimore', (39.28,76.62)]]

T=100000
mintemp=0
maxepoch=1000

initsol=initsol(cities)
paths=simulatedannealing(T,mintemp,maxepoch,initsol)
print("")
print("Initial Path = "+str(paths[0]))
print("")
print("Initial Fitness = "+str(f(paths[0])))
print("")
print("Best Path = "+str(paths[-1]))
print("")
print("Best Fitness = "+str(f(paths[-1])))
plotsol(paths)
