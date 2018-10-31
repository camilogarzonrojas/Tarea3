import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA

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

##
