import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

###Primera parte###
#Se cargan los dos archivos de datos
x1,y1=np.genfromtxt('signal.dat', unpack=True, delimiter=',')
x2,y2=np.genfromtxt('incompletos.dat', unpack=True, delimiter=',')
