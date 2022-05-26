#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
import sys
import numpy as np
from numpy import linalg as LA
import os
import re
import mmap
sys.path.insert(1, '../scripts')
from read_qeifcq import qe_phonon_info

# for qe-6.6

#dertermine file names of .epb? files
check_eph=False
if len(sys.argv) == 3:
  dir_epw = sys.argv[1]
  prefix = sys.argv[2]
elif len(sys.argv) == 4:
  dir_epw = sys.argv[1]
  prefix = sys.argv[2]
  dir_jdftx_ref = sys.argv[3]
  check_eph=True
else:
  exit('run this script with 2 or 3 argments - dir_epw prefix (dir_jdftx_ref)')
  
def fname_epb(i):
  return dir_epw + prefix + ".epb" + str(i)

#read e-ph matrix elements from multiple .epb? files
ifil = 1
qmesh = np.zeros(3).astype(np.int32)
nkstot = 0
while True:
  if not os.path.isfile(fname_epb(ifil)):
    break
  print("read "+fname_epb(ifil))
  f = open(fname_epb(ifil), 'rb')
  
  #in elphon_shuffle_wrap.f90, there is "WRITE(iuepb) nbnd, nks, nmodes, nqc1, nqc2, nqc3, xqc, dynq, epmatq"
  np.fromfile(f, np.int32, count=1)
  nbnd, nks, nmodes, qmesh[0], qmesh[1], qmesh[2] = np.fromfile(f, np.int32, count=6)
  nq = qmesh.prod()
  nkstot = nkstot + nks
  
  size_expected = 8 + 6*4 + 3*nq*8 + (nmodes*nmodes*nq + nbnd*nbnd*nks*nmodes*nq)*16
  fsize = os.path.getsize(fname_epb(ifil))
#  if fsize != size_expected:
#    exit(fname_epb(ip)+" file size is not right")
  
  xqc = np.fromfile(f, np.float64, count=3*nq).reshape(nq,3)
  
  dynq = np.fromfile(f, np.complex128, count=nmodes*nmodes*nq).reshape(nq,nmodes,nmodes) * 2 / 4 #2 - mass unit from Ry to a.u.; 4 - square of energy unit from Ry to Hartree
  dynq = np.transpose(dynq, (0,2,1))
  
  #read ep(q,m,k,i,j). i/j is related to k and k+q
  #then take transpose to get ep(q,m,k,j,i)
  eptmp = np.fromfile(f, np.complex128, count=nbnd*nbnd*nks*nmodes*nq).reshape(nq,nmodes,nks,nbnd,nbnd) / 2 / nq #1/2 - Ry to Hartree; 1/nq - prefactor for FFT
  np.fromfile(f, np.int32, count=1)
  eptmp = np.transpose(eptmp, (0,1,2,4,3))
  
  if ifil == 1:
    ep = eptmp
  else:
    ep = np.concatenate((ep, eptmp),axis=2)
  ifil = ifil + 1
ep.tofile('ep_epw.dat')
np.savetxt('xqc_epw.out',xqc)
if nkstot != nq:
  exit('nkstot is not nqc')

#help functions
def wrap(k, c=np.zeros(3)):
  r = k - c - np.floor(k - c + 0.5)
  r = np.where(abs(r-0.5)<1e-6, -0.5, r)
  return r+c
kcenter = np.array([0.5,0.5,0.5])
def k2ik(k):
  ktmp = wrap(k,kcenter)
  if k.ndim == 1:
    return int(np.round(ktmp[0]*qmesh[0]*qmesh[1]*qmesh[2] + ktmp[1]*qmesh[1]*qmesh[2] + ktmp[2]*qmesh[2]))
  else:
    return (np.round(ktmp[:,0]*qmesh[0]*qmesh[1]*qmesh[2] + ktmp[:,1]*qmesh[1]*qmesh[2] + ktmp[:,2]*qmesh[2])).astype(np.int32)
def k2ik_fft(k):
  ktmp = wrap(k,kcenter)
  if k.ndim == 1:
    return int(np.round(ktmp[2]*qmesh[2]*qmesh[1]*qmesh[0] + ktmp[1]*qmesh[1]*qmesh[0] + ktmp[0]*qmesh[0]))
  else:
    return (np.round(ktmp[:,2]*qmesh[2]*qmesh[1]*qmesh[0] + ktmp[:,1]*qmesh[1]*qmesh[0] + ktmp[:,0]*qmesh[0])).astype(np.int32)
def ik2k(ik):
  ikx = ik // (qmesh[1] * qmesh[2])
  iky = (ik // qmesh[2]) % qmesh[1]
  ikz = ik % qmesh[2]
  return np.array([ikx/qmesh[0], iky/qmesh[1], ikz/qmesh[2]]).reshape(3)

#Note: this script requires Yambo use the same q-grid as EPW
from netCDF4 import Dataset
import numpy as np

path='/data/users/cguo17/ex-ph-proj/MoS2_Example/'

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
    f=Dataset(path+'ndb-database/ndb.BS_diago_Q'+str(i+1),'r')
    BS_EIGENSTATES=f.variables['BS_EIGENSTATES'] # eigen-states
    A_exc[i]=np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,0] + 1j*np.array(BS_EIGENSTATES)[:,0:N_k*N_v*N_c,1]
    BS_TABLE=f.variables['BS_TABLE'] # BS_TABLE
    T_exc[i]=np.array(BS_TABLE)

print('Exciton wfc Table Set: \n' +str(T_exc[0]))
print('Exciton wfc Matrix Set Shape: \n' +str(A_exc.shape))

# read EPW el-ph matrix
g_elph=np.zeros((nq,nmodes,nks,N_v+N_c,N_v+N_c),dtype='complex128')
# grep the partial g_elph with the bands index covers BSE transition bands
g_elph=np.fromfile(path+'ep_epw.dat',dtype='complex128').reshape((nq,nmodes,nks,nbnd,nbnd))[:,:,:,N_e-N_v:N_e+N_c, N_e-N_v:N_e+N_c]
print('El-Ph Matrix Set Shape: \n' +str(g_elph.shape))

start_time = time.time()

num=3 #number of transitions from lowest excitation taken into account
n_ind_i = 62-1 #for test purpose choose the 1st exiton and it's nearby two
n_ind_f = n_ind_i+num-1

Gamma_1=np.zeros((N_Q,nq,N_k*N_v*N_c,N_k*N_v*N_c,nmodes),dtype='complex64')
Gamma_2=np.zeros((N_Q,nq,N_k*N_v*N_c,N_k*N_v*N_c,nmodes),dtype='complex64')
for Q_ind in range(0,N_Q): # exciton Q index
    for q_ind in range(0,nq): # phonon q index
         for m_ind in range(n_ind_i,n_ind_f): # bra-state exciton index        
             for n_ind in range(n_ind_i,n_ind_f): # ket-state exciton index
                for l_ind in range(0,nmodes): # phonon mode index
                    print('--- computing --- : (Q, q, m, n, l)= '+'('+str(Q_ind)+','+str(q_ind)+','+str(m_ind)+','+str(n_ind)+','+str(l_ind)+')')
                    print("--- %s seconds ---" % format((time.time() - start_time),'.2f'))
                    for k_ind in range(0,N_k): # exciton k index
                        if (T_exc[Q_ind,1,m_ind]==T_exc[Q_ind,1,n_ind]): # check the v index be the same, then loop the c, c' index
                            Gamma_1[Q_ind, q_ind, m_ind, n_ind, l_ind]+= (np.conj(A_exc[k2ik(ik2k(Q_ind)+ik2k(q_ind)), m_ind, k_ind])*(A_exc[Q_ind, n_ind, k_ind]) \
                                    *g_elph[q_ind, l_ind, k2ik(ik2k(k_ind)+ik2k(Q_ind)), T_exc[Q_ind,2,m_ind]-N_e, T_exc[Q_ind,2,n_ind]-N_e])
                        if (T_exc[Q_ind,2,m_ind]==T_exc[Q_ind,2,n_ind]): # check the c index be the same, then loop the v, v' index
                            Gamma_2[Q_ind, q_ind, m_ind, n_ind, l_ind]+= -(np.conj(A_exc[Q_ind, m_ind, k2ik(ik2k(k_ind)-ik2k(q_ind))])*(A_exc[Q_ind, n_ind, k_ind]) \
                                    *g_elph[q_ind, l_ind,k2ik(ik2k(k_ind)-ik2k(q_ind)), T_exc[Q_ind,2,m_ind]-N_e, T_exc[Q_ind,2,n_ind]-N_e])
print('Job Done!')
print("--- %s seconds ---" % format((time.time() - start_time),'.2f'))
Gamma = Gamma_1 + Gamma_2

