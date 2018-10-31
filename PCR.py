import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg

###Primera parte###

datos=pd.read_csv('WDBC.dat', header=None) # Se cargan los datos haciendo uso de pandas porque son de diferente tipo y numpy no los soport bien. Se hizo uso de un tutorial en internet para resolver este problema

###Segunda parte###

#Se convierten los datos en arreglos
diagnostico=np.array(datos[1])
mediciones=np.array(datos.iloc[:,2:])

#Se calcula la covarianza
delta=mediciones-np.mean(mediciones,0) #Se le resta a cada columna el promedio de dicha columna. En delta quedan guardados los valores
#Al multiplicar delta.T*delta se realizan todas las multiplicaciones [x-E(x)]*[y-E(y)] y se suman
#Al ya estar sumados, se divide entre n-1 para encontrar la covarianza
Cov=np.matmul(delta.T,delta)/(len(delta)-1)

print(Cov, '\n'*3, np.cov(mediciones.T))

###Tercera Parte###
EIG=linalg.eig(Cov)
print("Autovalores ", EIG[0])
print("Autovectores (Cada Columna)", EIG[1], "\n") #Se calculan los autovalores y los autovectores y se imprimen en pantalla

###Cuarta parte###
#SE ordenan los datos para extraer las componenetes principales
n=len(Cov)
indiceCP1=0 #Este es el indice de la primer componente principal
for i in range(n):
    if EIG[0][i] > EIG[0][indiceCP1]:
        indiceCP1 = i
        
indiceCP2 = 1 #Este el el indice de la segunda componente principal
for i in range(n):
    if EIG[0][i] > EIG[0][indiceCP2] and EIG[0][i] < EIG[0][indiceCP1]: #Se establece la condicion de quesea mayor a los otros y menor que la componente 1
        indiceCP2 = i

# Se extraen los autovectores de los autovalores mas grandes
vector_cp1=EIG[1][:,indiceCP1]
vector_cp2=EIG[1][:,indiceCP2]

print("Componente principal 1 ", vector_cp1)
print("Componente principal 2 ", vector_cp2)
print("Segun el componente principal 1, la variable mas importante es la columna", 2+np.argmax(np.abs(vector_cp1)), "del archivo de texto")
print("Segun el componente principal 2, la variable mas importante es la columna", 2+np.argmax(np.abs(vector_cp2)), "del archivo de texto")


###Cuarta parte###
#SE ordenan los datos para extraer las componenetes principales
n=len(Cov)
indiceCP1=0 #Este es el indice de la primer componente principal
for i in range(n):
    if EIG[0][i] > EIG[0][indiceCP1]:
        indiceCP1 = i
        
indiceCP2 = 1 #Este el el indice de la segunda componente principal
for i in range(n):
    if EIG[0][i] > EIG[0][indiceCP2] and EIG[0][i] < EIG[0][indiceCP1]: #Se establece la condicion de quesea mayor a los otros y menor que la componente 1
        indiceCP2 = i

# Se extraen los autovectores de los autovalores mas grandes
vector_cp1=EIG[1][:,indiceCP1]
vector_cp2=EIG[1][:,indiceCP2]

print("Componente principal 1 ", vector_cp1)
print("Componente principal 2 ", vector_cp2)
print("Segun el componente principal 1, la variable mas importante es la columna", 2+np.argmax(np.abs(vector_cp1)), "del archivo de texto")
print("Segun el componente principal 2, la variable mas importante es la columna", 2+np.argmax(np.abs(vector_cp2)), "del archivo de texto")
