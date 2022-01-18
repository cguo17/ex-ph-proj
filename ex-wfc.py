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

A_exc=np.zeros((N_Q,N_k*N_v*N_c,N_k*N_v*N_c),dtype='complex64')
T_exc=np.zeros((N_Q,5,N_k*N_v*N_c),dtype='int32')

# read Yambo finite Q BSE wavefunction
for i in range(0,N_Q):
    f=Dataset(path+'/ndb-database/ndb.BS_diago_Q'+str(i+1),'r')
    BS_EIGENSTATES=f.variables['BS_EIGENSTATES'] # eigen-states
    print('\nSize of BS matrix '+str(i+1)+' is: \n'+str((np.array(BS_EIGENSTATES).shape)))
    A_exc[i]=np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,0] + 1j*np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,1]
    BS_TABLE=f.variables['BS_TABLE'] # BS_TABLE
    T_exc[i]=np.array(BS_TABLE)

print('Exciton wfc Table Set: \n' +str(T_exc[0]))
print('Exciton wfc Matrix Set Shape: \n' +str(A_exc.shape))


print('Job Read Done!')
print("--- %s seconds ---" % (time.time() - start_time))
