
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import scipy as sc
def wood(X):

    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]

    summ = sum((
        100*(x1*x1 - x2)**2,
        (x1-1)**2,
        (x3-1)**2,
        90*(x3*x3 - x4)**2,
        10.1*((x2-1)**2 + (x4-1)**2),
        19.8*(x2-1)*(x4-1),
        ))
    
    return summ

def translate(DNAlist,bounds):
    realDNA = 0
    for i in range(len(DNAlist)):
        realDNA = realDNA * 2 + DNAlist[i]
    realDNA = realDNA / ((1 << len(DNAlist)) - 1)
    realDNA *= (bounds[1] - bounds[0])
    realDNA += bounds[0]
    return realDNA

class Individual(object):

    def __init__(self, dim = 4, DNA=None, mutate_prob=0.01,bounds=[-20,20]):
        if DNA is None:
            DNA_lenght = int(sc.ceil(sc.log2(bounds[1] - bounds[0] * pow(10,10))))
            self.DNA =  np.random.randint(2,size=(4, DNA_lenght))
        else: 
            self.DNA = DNA
            # Mutate
            for st in DNA:
                if mutate_prob > np.random.rand():
                    mutate_index = np.random.randint(len(self.DNA[st]) - 1)
                    self.DNA[st][mutate_index] = np.random.randint(2)

    def fitness(self,bounds=[-20,20]):
        translatedDNA = []
        for i in range(len(self.DNA)):
            translatedDNA.append(translate(self.DNA[i],bounds))
        return wood(translatedDNA)

class Population(object):

    def __init__(self, pop_size=10, mutate_prob=0.01, retain=0.2, random_retain=0.03):

        self.pop_size = pop_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.fitness_history = []
        self.parents = []
        self.done = False

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            self.individuals.append(Individual(DNA=None,mutate_prob=self.mutate_prob,bounds=bounds))

    def grade(self, generation=None,bounds=[-20,20]):
        fitness_sum = 0
        for x in self.individuals:
            fitness_sum += x.fitness()

        pop_fitness = fitness_sum / self.pop_size
        self.fitness_history.append(pop_fitness)

        # Set Done flag if we hit target
        if int(round(pop_fitness)) == 0:
            self.done = True

        if generation is not None:
            if generation % 10 == 0:
                print("Generation",generation,"Population fitness:", pop_fitness)

    def select_parents(self):

        # Sort individuals by fitness (we use reversed because in this case lower fintess is better)
        self.individuals = list(reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=True)))
        # Keep the fittest as parents for next gen
        retain_length = self.retain * len(self.individuals)
        self.parents = self.individuals[:int(retain_length)]

        # Randomly select some from unfittest and add to parents array
        unfittest = self.individuals[int(retain_length):]
        for unfit in unfittest:
            if self.random_retain > np.random.rand():
                self.parents.append(unfit)

    def breed(self):
 
        target_children_size = self.pop_size - len(self.parents)
        children = []
        if len(self.parents) > 0:
            while len(children) < target_children_size:
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                if father != mother:
                    child_numbers = [ random.choice(pixel_pair) for pixel_pair in zip(father.DNA, mother.DNA)]
                    child = Individual(child_numbers)
                    children.append(child)
            self.individuals = self.parents + children

    def evolve(self):
        # 1. Select fittest
        self.select_parents()
        # 2. Create children and new generation
        self.breed()
        # 3. Reset parents and children
        self.parents = []
        self.children = []

if __name__ == "__main__":
    
    pop_size = 1000
    mutate_prob = 0.01
    retain = 0.1
    random_retain = 0.03

    bounds = [-20,20]

    pop = Population(pop_size=pop_size, mutate_prob=mutate_prob, retain=retain, random_retain=random_retain)

    plot = True
    maxgeneratios = 1000
    
    for g in range(maxgeneratios):
        pop.grade(generation=g)
        pop.evolve()

        if pop.done:
            print("Finished at generation:", g, ", Population fitness:", pop.fitness_history[-1])
            print("Best Infividual")
            break

    # Plot fitness history
    if plot:
        print("Fitness history graph")
        plt.plot(np.arange(len(pop.fitness_history)), pop.fitness_history)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title('Fitness - pop_size {} mutate_prob {} retain {} random_retain {}'.format(pop_size, mutate_prob, retain, random_retain))
        plt.show()