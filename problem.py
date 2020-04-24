#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:53:53 2020

@author: jeremyjohnston
"""

import numpy as np


class Problem:
  def __init__(self, size_A, scen, partition=False, **kwargs):
    M = size_A[0]
    N = size_A[1]

    if 'N_part' in kwargs.keys():
      self.N_part = kwargs['N_part']

    self.partition = partition
    
    self.scen = scen
    # ptypes = ['gaussian', 'siso']
    # dgaussian = dict(admm=0.1,lista=0.1)
    # dsiso = dict(admm=0.02,lista=0.1)
    # self.lambda_init = dict(gaussian=dgaussian, siso=dsiso)

    if (scen == 'gaussian'):
      self.A = np.random.normal(size=(M,N))
      # normalize cols
      self.A = np.matmul(self.A,np.diag(1/np.sqrt(np.sum(self.A**2,axis=0))))
      
    elif (scen == 'siso'):
      m = np.arange(M)[:,None]
      n = np.arange(N)
      # self.A = np.cos(np.pi*m*n/N)
      self.A = np.exp(1j*2*np.pi*m*n/N)
    
    elif scen == 'siso_d':
      N = 8
      Ngl = 5
      Ngk = 3
      self.A = np.zeros((N,Ngk*Ngl),dtype=np.complex128)
      for l in range(Ngl):
        for k in range(Ngk):
          # self.A[:, l*Ngk + k] = np.cos(np.pi*((l/Ngl)*np.arange(N)+(k/Ngk)*np.arange(N)**2))
          self.A[:, l*Ngk + k] = np.exp(1j*np.pi*((l/Ngl)*np.arange(N)+(k/Ngk)*np.arange(N)**2))

    elif (scen == 'mimo_real'):
      Np = 3
      Ntx = 2
      Nrx = 2

      Ngl = 5 # range grid
      Ngk = 5 # DoA grid

      ngl = np.arange(Ngl)
      ngk = np.arange(Ngk)

      v = np.cos(np.pi*np.outer(np.arange(Np),ngl)/Ngl)
      hrx = np.cos(np.pi*np.outer(np.arange(Nrx),ngk)/Ngk)
      htx = np.cos(np.pi*np.outer(np.arange(Ntx),ngk)/Ngk)
      h = khatri_rao(hrx,htx)
      self.A = np.kron(h,v)

    elif (scen == 'mimo_r'):
      # Np = 2
      # Ntx = 1
      # Nrx = 2

      # Ngl = 3 # range grid
      # Ngk = 3 # DoA grid

      Np = 4;
      Ntx = 2;
      Nrx = 1;
      Ngl = 4;
      Ngk = 4;

      ngl = np.arange(Ngl)
      ngk = np.arange(Ngk)
      nn = np.arange(Np)
      nrx = np.arange(Nrx)
      ntx = np.arange(Ntx)

      v = np.exp(1j*2*np.pi*np.outer(nn,ngl)/Ngl)
      hrx = np.exp(1j*2*np.pi*np.outer(nrx,ngk)/Ngk)
      htx = np.exp(1j*2*np.pi*np.outer(ntx,ngk)/Ngk)
      h = khatri_rao(hrx,htx)
      A = np.kron(h,v)
      self.A=A
      # Atop = np.concatenate((np.real(A),-np.imag(A)),axis=1)
      # Abot = np.concatenate((np.imag(A),np.real(A)),axis=1)
      # self.A = np.concatenate((Atop,Abot),axis=0)

    elif (scen == 'mimo'):
      # Np = 2
      # Ntx = 1
      # Nrx = 2

      # Ngl = 3 # range grid
      # Ngk = 3 # DoA grid

      Ntx = 4;
      Nrx = 2;

      Ngk = 16;


      ngk = np.arange(Ngk)
      nrx = np.arange(Nrx)
      ntx = np.arange(Ntx)

      hrx = np.exp(1j*2*np.pi*np.outer(nrx,ngk)/Ngk)
      htx = np.exp(1j*2*np.pi*np.outer(ntx,ngk)/Ngk)
      self.A=khatri_rao(hrx,htx)
      # Atop = np.concatenate((np.real(A),-np.imag(A)),axis=1)
      # Abot = np.concatenate((np.imag(A),np.real(A)),axis=1)
      # self.A = np.concatenate((Atop,Abot),axis=0)
      #   

    elif (scen == 'mimo_d'):
      
      N = 4;
      Ntx = 2;
      Nrx = 1;
      # sampling grid
      M = N*Ntx*Nrx;
      Ngm = 2; 
      Ngl = 3;
      Ngk = 3;

      nn = np.arange(N)
      nrx = np.arange(Nrx)
      ntx = np.arange(Ntx)

      self.A = np.zeros((M,Ngm*Ngl*Ngk),dtype=np.complex128);
      for l in range(Ngl):
          for k in range(Ngk):
              for m in range(Ngm):
                  v = np.exp(1j*np.pi*((l/Ngl)*nn.T + (m/Ngm)*nn.T**2));
                  H = np.outer(np.exp(1j*2*np.pi*(k/Ngk)*nrx),np.exp(1j*2*np.pi*(k/Ngk)*ntx))
                  h = H.ravel();
                  self.A[:, l*Ngm*Ngk + k*Ngm + m] = np.kron(h,v);
    
    elif scen == 'yin':
      # dimensions of the sparse signal x
      n = 8
      # dimension of the compressed signal y
      m = 4
      # sparsity : non-zero values over n
      sparsity = 2

      # training and testing number:
      train_number = 50000
      valid_number = 100
      # number of training iterations (epochs)
      numIter = 500

      # w state of randomness
      # rng = np.random.RandomState(23)


      # create the 1:N sample
      sample = np.array(list(range(m))).reshape(m, 1)
      # create the frequency grid
      f_grid = 1/n*np.array(list(range(n))).reshape(1, n)

      # generate the dictionary, without the sampling matrix Phi
      C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
      A = C.copy()

      # generate the dictionary with the sampling matrix Phi
      C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
      phi = np.ones([m])
      p = 0.8 # sparsity level
      Ns = int(np.floor((1-p)*m)) # the sampling number, note that should be int
      index = rd.sample(range(m),Ns)
      phi[index] = 0
      A = C.copy()
      for i in range(n):
        A[index, i] = 0
        # print(A[index, i])    
      self.A = A

    if self.partition == True:
      self.A = np.concatenate((self.A, np.eye(np.shape(self.A)[0])),axis=1)

    return

  
    
  def size(self, dim=None):
    if dim is None:
      return self.A.shape
    else:
      return self.A.shape[dim]

def khatri_rao(a, b):
    c = a[...,:,np.newaxis,:] * b[...,np.newaxis,:,:]
    # collapse the first two axes
    return c.reshape((-1,) + c.shape[2:])

#L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
# class GaussianProblem(Problem):