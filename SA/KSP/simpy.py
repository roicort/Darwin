"""

Simulated Annealing (STP)

Developped by Rodrigo Cortez
ENES Morelia, 2018

"""

import scipy as sc
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import copy

def chooseneighboor(sol,c):
    global listaobjetos
    while True:
        a=random.randint(0,len(sol)-1)
        b=random.randint(0,len(listaobjetos)-1)
        sol[a],listaobjetos[b] = listaobjetos[b],sol[a]
        if weight(sol) < c:
            return sol

def calctemp(T):
    return (T*0.9999)

def weight(sol):
    ww=0
    for i in sol:
        ww+=i[0]
    return ww

def f(sol):
    ff=0
    for i in sol:
        ff+=i[1]
    return ff

def getinitsol(c):
    global listaobjetos
    sol = []
    solweight = 0
    random.shuffle(listaobjetos)
    for o in listaobjetos:
        if solweight+o[0]<c:
            sol.append(o)
            listaobjetos.remove(o)
    return sol

def simulatedannealing(T,mintemp,maxepoch,initsol,c):
    t=0
    sol = initsol
    while T > mintemp and t < maxepoch:
        j=chooseneighboor(sol,c) #elegir j
        if f(j) > f(sol):
            sol=j
        if random.randint(0,1) < sc.exp((f(j)-f(sol)/T)): #if random.randint(0,1) < sc.exp((-f(j)-f(sol)/T)):
            sol=j
        t+=1
        T = calctemp(T)
    return sol

T=100000000000000
mintemp=0.00000000000000000000000000000000000000000000000000000000000001
maxepoch=100000

problem = "p08_"

capacity = pd.read_csv(problem+"c.txt",header=None,dtype='float')
c = capacity[0][0]
print("")
print("KS Capacity = "+ str(c))
w = pd.read_csv(problem+"w.txt",header=None,dtype='float')
p = pd.read_csv(problem+"p.txt",header=None,dtype='float')
df = np.array(pd.concat([w,p],axis=1))

listaobjetos = []
for i in df:
    listaobjetos.append((i[0],i[1]))
initsol = list(getinitsol(c))
print("")
print("Initsol = " +str(initsol) +" "+ "Value = " +str(f(initsol)) +" "+"Weight = "+str(weight(initsol)))
print("")
bestsol = simulatedannealing(T,mintemp,maxepoch,initsol,c)
print("Bestsol = " +str(bestsol) +" "+ "Value = " +str(f(bestsol)) +" "+"Weight = "+str(weight(bestsol)))
print("")