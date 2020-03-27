#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:53:53 2020

@author: jeremyjohnston
"""

import numpy as np
import sympy


class Problem:
  def __init__(self, size_A, scen, **kwargs):
    M = size_A[0]
    N = size_A[1]
    
    self.scen = scen
    # ptypes = ['gaussian', 'siso']
    # dgaussian = dict(admm=0.1,lista=0.1)
    # dsiso = dict(admm=0.02,lista=0.1)
    # self.lambda_init = dict(gaussian=dgaussian, siso=dsiso)

    if (scen == 'gaussian'):
      self.A = np.random.rand(M, N)
      # normalize cols
      self.A = np.matmul(self.A,np.diag(1/np.sqrt(np.sum(self.A**2,axis=0))))
      # if 'lambda_init' in kwargs.keys():
      #   for k in kwargs['lambda_init'].keys():
      #     self.lambda_init[k] = kwargs['lambda_init'][k]
      # self.rho_init = 1.
      # self.alph_init = 1.
      
    elif (scen == 'siso'):
      m = np.arange(M)[:,None]
      n = np.arange(N)
      self.A = np.cos(np.pi*m*n/N)
    
    elif scen == 'siso_d':
      N = 120
      Ngl = 40
      Ngk = 6
      self.A = np.zeros((N,Ngk*Ngl))
      for l in range(Ngl):
        for k in range(Ngk):
          self.A[:, l*Ngk + k] = np.cos(np.pi*((l/Ngl)*np.arange(N)+(k/Ngk)*np.arange(N)**2))
    elif (scen == 'mimo'):
      Np = 4
      Ntx = 2
      Nrx = 2

      Ngl = 8 # range grid
      Ngk = 4 # DoA grid

      ngl = np.arange(Ngl)
      ngk = np.arange(Ngk)

      v = np.cos(np.pi*np.outer(np.arange(Np),ngl)/Ngl)
      hrx = np.cos(np.pi*np.outer(np.arange(Nrx),ngk)/Ngk)
      htx = np.cos(np.pi*np.outer(np.arange(Ntx),ngk)/Ngk)
      h = khatri_rao(hrx,htx)
      self.A = np.kron(h,v)

    elif (scen == 'mimo_c'):
      Np = 4
      Ntx = 2
      Nrx = 2

      Ngl = 8 # range grid
      Ngk = 4 # DoA grid

      ngl = np.arange(Ngl)
      ngk = np.arange(Ngk)

      v = np.exp(1j*2*np.outer(np.arange(Np),ngl)/Ngl)
      hrx = np.exp(1j*2*np.pi*np.outer(np.arange(Nrx),ngk)/Ngk)
      htx = np.exp(1j*2*np.outer(np.arange(Ntx),ngk)/Ngk)
      h = khatri_rao(hrx,htx)
      A = np.kron(h,v)
      Atop = np.concatenate((np.real(A),-np.imag(A)),axis=1)
      Abot = np.concatenate((np.imag(A),np.real(A)),axis=1)
      self.A = np.concatenate((Atop,Abot),axis=0)
      
    elif (scen == 'mimo_d'):
      
      N = 50;
      Ntx = 4;
      Nrx = 4;
      # sampling grid
      M = N*Ntx*Nrx;
      Ngm = 3; 
      Ngl = 20;
      Ngk = 5;
      self.A = np.zeros((M,Ngm*Ngl*Ngk));
      for l in range(Ngl):
          for k in range(Ngk):
              for m in range(Ngm):
                  v = np.cos(0.5*np.pi*((l/Ngl)*np.arange(N).T + (m/Ngm)*np.arange(N).T**2));
                  H = np.outer(np.cos(np.pi*(k/Ngk)*np.arange(Nrx)),np.cos(np.pi*(k/Ngk)*np.arange(Ntx)))
                  h = H.ravel();
                  self.A[:, l*Ngm*Ngk + k*Ngm + m] = np.kron(h,v);
              

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