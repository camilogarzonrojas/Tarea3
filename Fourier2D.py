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
