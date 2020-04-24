import problem
import admm
import admm_interference as admm_i
import lista
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import time
import load_net

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def gen_samples(A,N,density=0.1,normalize=False,comp=False, ones=True):
  N = int(N)
  n1,n2 = A.shape

  # x = sp.sparse.random(n2,N,density,dtype=np.float32).A
  # y = np.matmul(A,x)

  # x = x.T
  # y = y.T
  if comp==True:
    x = np.zeros((N,n2//2),dtype=np.complex64)
    for i in range(N):
      r = sp.sparse.random(1,n2//2,density,dtype=np.float32,format='array')[0]
      x[i] = np.exp(1j*2*np.pi*r)
    x = np.concatenate((np.real(x), np.imag(x)),axis=1)
  else:
    x = np.zeros((N,n2))
    for i in range(N):
      x[i] = sp.sparse.random(1,n2,density,dtype=np.float32,format='array')[0]

  y = np.matmul(A,x.T).T

  # breakpoint()


  # x = sp.sparse.random(n2,N,density,dtype=np.float32,format='array').T
  # y = np.matmul(x,A.T)
  # breakpoint()
  # if comp == True:
  #   for i in range(N):
  #     t = sp.sparse.random(n2,1,density,dtype=np.complex64).A.T[0]
  #     x[i] = np.concatenate((np.real(t),np.imag(t)))
  #     if normalize == True:
  #       x[i] = x[i]/np.sqrt(np.sum(x[i]**2))
  #     if ones == True:
  #       x[i] = np.divide(x[i],x[i],out=np.zeros_like(x[i]),where=x[i]!=0)
  #     y[i] = np.matmul(A,x[i]).T
  # else:
  

  
  return x,y

if __name__ == '__main__':
  p = problem.Problem((20,40), 'mimo_c')
  # sparsity density
  d = 0.05
  x,y = gen_samples(p.A, 1e5, density=0.25, normalize=False, ones=False, comp=True)
  np.save('./trainingdata/1e5_mimo_c_4x8_x',x)
  np.save('./trainingdata/1e5_mimo_c_4x8_y',y)