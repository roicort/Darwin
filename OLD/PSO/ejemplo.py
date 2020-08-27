#libreria de python que permite leer archivos con extension .json
import json
#aqui se pone el nombre del archivo json
data = json.loads(open('matrices.json').read())
# aqui cargamos las matrices del archivo
matrizTiempo=data[0]["matrizTiempo"]
matrizDistancia=data[0]["matrizDistancia"]
pdvs=data[0]["IDName"]
# aqui imprimimos las matrices
# descomentar las sig 2 lineas para ver las matrices
# print(matrizDistancia)
# print(matrizTiempo)
# las matrices tienen la siguiente estructura
"""
        pdv1    pdv2    pdv3    ...     pdvN
pdv1     0       223    322     ...     123
pdv2    674       0     145     ...     429
pdv3    844      256     0      ...     757
...
pdvN    769      168    655     ...     0
"""
# para acceder a la distancia del primer punto al segundo punto del sector
# deberia ser de la sig manera matrizDistancia[0][1]
# se aplica lo mismo para la matriz de tiempo
