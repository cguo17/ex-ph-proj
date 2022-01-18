#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np

path='/data/users/cguo17/ex-ph-proj'

N_e= 26 # number of electron
N_Q= 6*6*1 # number of Yamnbo finite Q BSE Q-point
N_k= N_Q # number of kpoints in Yambo
N_v= 4 # number of valence bands in BSE
N_c= 4 # number of condunction bands in BSE
nq= 6*6*1 # number of EPW q-point
nmodes= 3*3 # number of phonon modes
nks= 6*6*1 # number of kpoints in EPW
nbnd= 50 # number of bands in EPW

import time

A_exc=np.zeros((N_Q,N_k*N_v*N_c,N_k*N_v*N_c),dtype='complex64')
T_exc=np.zeros((N_Q,5,N_k*N_v*N_c),dtype='int32')

# read Yambo finite Q BSE wavefunction
for i in range(0,N_Q):
    f=Dataset(path+'/ndb-database/ndb.BS_diago_Q'+str(i+1),'r')
    BS_EIGENSTATES=f.variables['BS_EIGENSTATES'] # eigen-states
    A_exc[i]=np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,0] + 1j*np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,1]
    BS_TABLE=f.variables['BS_TABLE'] # BS_TABLE
    T_exc[i]=np.array(BS_TABLE)

print('Exciton wfc Table Set: \n' +str(T_exc[0]))
print('Exciton wfc Matrix Set Shape: \n' +str(A_exc.shape))

# read EPW el-ph matrix
g_elph=np.zeros((nq,nmodes,nks,N_v+N_c,N_v+N_c),dtype='complex128')
g_elph=np.fromfile(path+'/epw-database/ep_epw.dat',dtype='complex128').reshape((nq,nmodes,nks,nbnd,nbnd))[:,:,:,N_e-N_v:N_e+N_c, N_e-N_v:N_e+N_c]
print('El-Ph Matrix Set Shape: \n' +str(g_elph.shape))

start_time = time.time()

num=8
n_ind_f = N_k*N_v*N_c
n_ind_i = N_k*N_v*N_c-num

Gamma_1=np.zeros((N_Q,nq,N_k*N_v*N_c,N_k*N_v*N_c,nmodes),dtype='complex64')
Gamma_2=np.zeros((N_Q,nq,N_k*N_v*N_c,N_k*N_v*N_c,nmodes),dtype='complex64')
for Q_ind in range(0,N_Q): # exciton Q index
    for q_ind in range(0,nq): # phonon q index
         for m_ind in range(n_ind_i,n_ind_f): # bra-state exciton index        
             for n_ind in range(n_ind_i,n_ind_f): # bra-state exciton index
                for l_ind in range(0,nmodes): # phonon mode index
                    print('--- computing --- : (Q, q, m, n, l)= '+'('+str(Q_ind)+','+str(q_ind)+','+str(m_ind)+','+str(n_ind)+','+str(l_ind)+')')
                    print("--- %s seconds ---" % (time.time() - start_time))
                    for k_ind in range(0,N_k): # exciton k index
                        if (T_exc[Q_ind,1,m_ind]==T_exc[Q_ind,1,n_ind]): # check the v index be the same, then loop the c, c' index
                            Gamma_1[Q_ind, q_ind, m_ind, n_ind, l_ind]+= (np.conj(A_exc[(Q_ind+q_ind)%N_k, m_ind, k_ind])*(A_exc[Q_ind, n_ind, k_ind]) \
                                    *g_elph[q_ind, l_ind, (k_ind+Q_ind)%N_k, T_exc[Q_ind,2,m_ind]-N_e, T_exc[Q_ind,2,n_ind]-N_e])
                        if (T_exc[Q_ind,2,m_ind]==T_exc[Q_ind,2,n_ind]): # check the c index be the same, then loop the v, v' index
                            Gamma_2[Q_ind, q_ind, m_ind, n_ind, l_ind]+= -(np.conj(A_exc[Q_ind, m_ind, (k_ind-q_ind)%N_k])*(A_exc[Q_ind, n_ind, k_ind]) \
                                    *g_elph[q_ind, l_ind,(k_ind-q_ind)%N_k, T_exc[Q_ind,2,m_ind]-N_e, T_exc[Q_ind,2,n_ind]-N_e])
print('Job Done!')
print("--- %s seconds ---" % (time.time() - start_time))
Gamma = Gamma_1 + Gamma_2
