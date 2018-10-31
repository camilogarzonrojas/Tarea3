import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2, fft2
from scipy.ndimage import imread

###Primera parte###
arbol = imread('arbol.PNG', flatten=True)
print("Cargar imagen del arbol", np.shape(arbol)) #Se cargan los datos
