
import numpy as np
import scipy as sc
from tests import ackley, ackleybounds
from copy import deepcopy

def F(X):
    return ackley(X)
    
class individual:

    def __init__(self,genes):
        self.DNA = np.random.randint(0, 2, genes)
        self.F = None

    def translateDNA(self):
        real = self.DNA 
        return real

    def mutate(self):
        if np.random.rand() < mutation_rate:
            point = np.random.randint(len(self.DNA))
            self.DNA[point] = 1 - self.DNA[point]

    def __repr__(self):
        return "DNA "+str(self.DNA)+" Fitness"+" = "+str(self.F)

def select(population,selected=2):
    maxx = sum(individual.F for individual in population)
    pick = np.random.uniform(0, maxx)
    apt = 0
    parents=[]
    for individual in population:
        apt += individual.F
        if apt > pick:
            parents.append(individual)
        if len(parents) == selected:
            return parents[0], parents[1]

def cross(parent1, parent2,cross_rate):
    ind1 = deepcopy(parent1)
    ind2 = deepcopy(parent2)
    if np.random.rand() < cross_rate:
        size = len(ind1.DNA)
        cxpoint1 = sc.random.randint(1, size)
        cxpoint2 = sc.random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1.DNA[cxpoint1:cxpoint2], ind2.DNA[cxpoint1:cxpoint2] = ind2.DNA[cxpoint1:cxpoint2], ind1.DNA[cxpoint1:cxpoint2]
        return ind1, ind2

def updatefitness(population):
    for individual in population:
        individual.F = F(individual.translateDNA())
    return population

generations = 100
population_size = 10

bounds = [-30,30]
genes = abs(bounds[0]-bounds[1])

cross_rate = 0.7
mutation_rate = 0.05

population = [individual(genes) for i in range(population_size)]
population = updatefitness(population)
population.sort(key=lambda objectt: objectt.F)
bestindividual=population[0]

for _ in range(generations):
    parents = select(population)
    offspring = []

    while len(offspring) < len(population):

        parent1,parent2 = select(population)

        child1,child2 = cross(parent1,parent2,cross_rate)

        offspring.append(child1)
        offspring.append(child2)

    offspring = updatefitness(offspring)
    offspring.sort(key=lambda objectt: objectt.F)
    bestindividual=offspring[0]
    population = offspring

print(bestindividual)