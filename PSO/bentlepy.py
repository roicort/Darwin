import pygmo as pg

points = [[8, 9, 2], [9, 3, 6], [9, 0, 2], [3, 2, 3], [3, 8, 0], [8, 2, 0], [0, 9, 4], [2, 3, 0], [4, 1, 0], [2, 5, 3], [7, 8, 0]]

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

def nondomiatedsorting(population):
    nondominated = []
    for A in population:
        for B in population:
            d = dominance(A,B)
            if d == -1:
                break
        if d == 1:
            nondominated.append(A)
        d = 0
    return nondominated

def bentleynondomiatedsorting(population):
    nondominated = [population[0]]
    for p in range(len(population)):
        popaux = True
        pn=0
        while pn < len(nondominated):
            aux=None
            if dominance(nondominated[pn],population[p]) == 1:
                aux = nondominated[pn].copy()
                nondominated.pop(pn)
                nondominated.insert(0,aux)
                popaux = False
                break
            if dominance(population[p],nondominated[pn]) == 1:
                nondominated.pop(pn)
            else:
                pn+=1
        if popaux:
            nondominated.append(population[p])
    return nondominated[1:]

bentley = bentleynondomiatedsorting(points)
print(bentley)