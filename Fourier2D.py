import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2, fft2
from scipy.ndimage import imread

###Primera parte###
arbol=imread('arbol.PNG', flatten=True)
print("Cargar imagen del arbol", np.shape(arbol)) #Se cargan los datos


###Segunda parte###
arbol_F=fft2(arbol)
print("Transformada de Fourier del arbol") #Transformada de Fourier usando paquete de scipy

fig,ax=plt.subplots()
ax.imshow(np.abs(arbol_F),vmin=1000,vmax=5000)
ax.set_xlabel(['x'])
ax.set_ylabel(['y'])
fig.savefig('GarzonCamilo_FT2D.pdf') #Se grafica la transformada de Fourier


###Tercera parte###
m,n=np.shape(arbol_F)
Filtrado = np.copy(arbol_F) #Se comienza a filtrar

for i in range(m):
    for j in range(n):
       
        if (0<i and i<20) and (20<j and j<40):
            Filtrado[i,j]=0
        if (60<i and i<80) and (60<j and j<80):
            Filtrado[i,j]=0
        if (180<i and i<200) and (180<j and j<200):
            Filtrado[i,j]=0
        if (240<i and i<250) and (220<j and j<240):
            Filtrado[i,j]=0   #Se dejan en cero las amplitudes de las frecuencias que generan el ruido periodico haciendo un doble recorrido


###Cuarta parte###
fig,ax=plt.subplots()
ax.imshow(np.log(np.abs(Filtrado)))
ax.set_xlabel(['x'])
ax.set_ylabel(['y'])
fig.savefig('GarzonCamilo_FT2D_filtrada.pdf') #Se grafica la transformada de Fourier en escala lognorm

