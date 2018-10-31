import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d

###Primera parte###
#Se cargan los dos archivos de datos
x1,y1=np.genfromtxt('signal.dat', unpack=True, delimiter=',')
x2,y2=np.genfromtxt('incompletos.dat', unpack=True, delimiter=',')


###Tercera parte###
def fourier(U):     #Se define la transformada discreta de Fourier
    N=len(U)
    fase=[]
    modulo=[]
    real=[]
    imag=[]
    
    for ki in range(N):
        UTk=0j+0 #Se inicializa en 0
        for ni in range(N):
            UTk=UTk+U[ni]*np.exp(1j*2*np.pi*ki*ni/N)
        
        #Se extrae la fase y el modulo de cada numero
        fase.append(np.arctan(UTk.imag/UTk.real))
        modulo.append(np.abs(UTk))
        real.append(UTk.real)
        imag.append(UTk.imag)
    
    return np.array(modulo), np.array(fase), np.array(real), np.array(imag)

def ff_frecuencias(Fnyq, n):
    f0=Fnyq/n #Esta es la frecuencia base
    return np.concatenate((np.linspace(f0,Fnyq,n), np.linspace(-Fnyq,-f0,n)))
    
#Se calculan las variables de Fourier    
modulo, fase, real, imag=fourier(y1)
frecuencias=ff_frecuencias(1.0/(x1[1]-x1[0])*(1.0/2.0), int(len(x1)/2))


###Grafica de la segunda parte###
fig, ax=plt.subplots()
ax.plot(x1, y1, c='black')
ax.set_xlabel(['x'])
ax.set_ylabel(['y'])
fig.savefig('GarzonCamilo_signal.pdf')



###Cuarta parte###
fig, ax=plt.subplots()
ax.plot(frecuencias, modulo)
ax.set_xlabel(['Espectro de frecuencias'])
ax.set_ylabel(['Modulo'])
fig.savefig('GarzonCamilo_TF.pdf')
