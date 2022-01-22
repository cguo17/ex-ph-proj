#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np

path='/data/users/cguo17/yambo-ph/bulk_hBN_electron_phonon_nk6x6x6'

N_e= 16 # number of electron
N_Q= 34 # number of Yamnbo finite Q BSE Q-point
N_k= 6*6*6 # number of kpoints in Yambo
N_v= 4 # number of valence bands in BSE
N_c= 4 # number of condunction bands in BSE
nq= 6*6*6 # number of EPW q-point
nmodes= 3*2 # number of phonon modes
nks= 6*6*1 # number of kpoints in EPW
nbnd= 4+4 # number of bands in EPW

import time

start_time = time.time()
arr_1=np.linspace(1, 4, num=4)
arr_2=np.linspace(8, 5, num=4)
arr_3=np.linspace(9, 12, num=4)
arr_4=np.linspace(16, 13, num=4)
arr=np.concatenate((arr_1, arr_2, arr_3, arr_4), axis=None)
arr_term=np.asarray([1,29,58,86,186,196,207,217,371,399,428,456,556,584,613,641])
A_exc=np.zeros((N_Q,N_k*N_v*N_c,N_k*N_v*N_c),dtype='complex64')
T_exc=np.zeros((N_Q,5,N_k*N_v*N_c),dtype='int32')
E_exc=np.zeros((N_Q,N_k*N_v*N_c,2),dtype='float32')
# read Yambo finite Q BSE wavefunction
for i in range(0,16):
    f=Dataset(path+'/ndb-database/ndb.BS_diago_Q'+str(i+1),'r')
    BS_EIGENSTATES=f.variables['BS_EIGENSTATES'] # eigen-states
    print('\nSize of BS matrix '+str(i+1)+' is: \n'+str((np.array(BS_EIGENSTATES).shape)))
    A_exc[i]=np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,0] + 1j*np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,1]
    BS_TABLE=f.variables['BS_TABLE'] # BS_TABLE
    T_exc[i]=np.array(BS_TABLE)
    print('Exciton wfc Table Set: \n' +str(T_exc[i]))
    print('Exciton wfc Table Set: \n' +str(T_exc[i].shape))
    BS_Energies=f.variables['BS_Energies'] # BS_Energies
    E_exc[i]=np.array(BS_Energies)
    print('Exciton BS_Energies Table Set: \n' +str(E_exc[i]))
    print('Exciton BS_Energies Table Set Max,Min: \n' +str(E_exc[i,:,0].max())+str(',')+str(E_exc[i,:,0].min()))
    print('Exciton BS_Energies Table Set: \n' +str(E_exc[i].shape))

print('Exciton wfc Matrix Set Shape: \n' +str(A_exc.shape))
print('Exciton wfc Table Set: \n' +str(T_exc[0]))

print('Job Read Done!')
print("--- %s seconds ---" % (time.time() - start_time))

import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks
#from brokenaxes import brokenaxes

plt.rc('font', family='arial')

bohr=1.88973
hartree=27.2114
smearing=0.02
L_0=20.
c=137.
t_ps=2.42*(10.**(-17.))*(10.**(12.))
DEG2RAD=1./57.2958

o_exc_interp=np.loadtxt('o.excitons_interpolated_01')
d=6.6*bohr*2
t=1./(d/(4.*np.pi))
for i in range(1,8):
    plt.plot(o_exc_interp[:,0], o_exc_interp[:,i], linewidth=2.0)

for j in range(0,16):
    for k in range(0,8):
        plt.scatter(o_exc_interp[np.int(arr_term[j])-1,0],E_exc[np.int(arr[j])-1,k,0]*hartree)
plt.xlim(0.,o_exc_interp[-1,0])
#plt.ylim(4.5,6.5)
plt.xlabel(r'|q| (a.u.)')
plt.ylabel(r'E (eV)')
plt.legend(loc='upper left', markerscale=0.75, fontsize=13)
plt.savefig('o_interp_exc.png',  dpi= 300, bbox_inches='tight')
#plt.show()
