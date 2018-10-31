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



###Quinta parte###
#Se encuentran los indices de los dos primeros picos
i_max = np.argmax(modulo)
print("La frecuencia principal se da en ", frecuencias[i_max])
copia=modulo
copia[i_max]=0
i_max2=np.argmax(copia)
copia[i_max2]=0
i_max2=np.argmax(copia)
print("La segunda frecuencia principal se da en ", abs(frecuencias[i_max2]))



###Sexta parte###
filtro=real+1j*imag #Se filtra la senal

for i in range(len(frecuencias)):
    if np.abs(frecuencias[i])>1000: #Si la frecuencia es mayor a 1000
        filtro[i] = filtro[i]/10 #Se disminuye la intensidad

y1_filtrado=ifft(filtro).real #Se devuelve al dominio del tiempo

fig,ax=plt.subplots()
ax.plot(x1, y1_filtrado)
ax.set_xlabel(['tiempo'])
fig.savefig('GarzonCamilo_filtrada.pdf') #Se hace la grafica


###Septima parte###
print("Los tres primeros deltas de tiempo de los datos incompletos son ", x2[1:4]-x2[0:3], ". Como estos valores no son constantes, por la definicion de la DFT, no se puede hacer la trasnformada de Fourier discreta")


###Octava parte###
x2linspace=np.linspace(0.00039063, 0.02851562, 512) #Este es el x para hacer interpolacion


y2interpolado=[interp1d(x2,y2,'cuadratica')(x2linspace), interp1d(x2,y2,'cubica')(x2linspace)] #Se interpolan con splines cubicos y cuadraticos


mq, fq, rq, iq = fourier(y2interpolado[0])
mc, fc, rc, ic = fourier(y2interpolado[1]) #Se calculan las variables de Fourier y se guardan en las listas 

modulo_inter=[mq,mc]
fase_inter=[fq,fc]
real_inter=[rq,rc]
imag_inter=[iq,ic]


frecuencias_interpolacion=ff_frecuencias( 1.0/(x2linspace[1]-x2linspace[0])*(1.0/2.0), int(len(x2linspace)/2) ) #Se calculan las frecuencias


###Novena parte###
fig,ax=plt.subplots(3,1,figsize=(8,8))
ax[0].plot(frecuencias,modulo)
ax[1].plot(frecuencias_interpolacion,modulo_inter[0])
ax[2].plot(frecuencias_interpolacion,modulo_inter[1])

ax[2].set_xlabel(['Espectro de frecuencias'])
ax[0].set_xlabel(['Senal base'])
ax[0].set_xlabel(['Interpolacion cuadrada'])
ax[0].set_xlabel(['Interpolacion cubica'])
fig.savefig('GarzonCamilo_TF_interpola.pdf') #Se grafican los tres espectros


###Decima parte###


###Undecima parte###
filtro_inter=np.array(real_inter)+1j*np.array(imag_inter) #Se filtra la senal

for i in range(len(frecuencias_interpolacion)):
    if np.abs(frecuencias_interpolacion[i])>1000: #Si la frecuencia es mayor a 1000
        filtro_inter[:,i]=filtro_inter[:,i]/10 #Se disminuye la intensidad
        
y_inter_filtrado=ifft(filtro_inter).real #Se devuelve al dominio


fig,ax=plt.subplots(2,1, figsize=(8,8))
ax[0].plot(x1, y1_filtrado)
ax[0].plot(x2linspace, y_inter_filtrado[0,:])
ax[0].plot(x2linspace, y_inter_filtrado[1,:])
ax[0].set_ylabel('Filtro a 1000 Hz') #Se grafican las señales a 1000 Hz y haer hecho transformada inversa



filtro_inter=np.array(real_inter)+1j*np.array(imag_inter)
filtro=real+1j*imag   #Se filtra la senal

for i in range(len(frecuencias_interpolacion)):
    if np.abs(frecuencias_interpolacion[i])>500: #Si la frecuencia es mayor a 500
        filtro_inter[:,i]=filtro_inter[:,i]/10 #Se disminuye la intensidad
        filtro[i]=filtro[i]/10 #Se disminuye la intensidad
        
y_inter_filtrado=ifft(filtro_inter).real #Se devuelve al dominio
y1_filtrado=ifft(filtro).real #Se devuelve al dominio


###Duodecima parte###
ax[1].plot(x1, y1_filtrado)
ax[1].plot(x2linspace, y_inter_filtrado[0,:])
ax[1].plot(x2linspace, y_inter_filtrado[1,:])
ax[1].set_ylabel('Filtro a 500 Hz')
ax[1].legend(['o','i2','i3'])
ax[0].legend(['o','i2','i3'])

fig.savefig('GarzonCamilo_2Filtros.pdf') #Se grafican las señales a 500 Hz
