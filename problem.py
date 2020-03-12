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
      self.A = np.sqrt(1/M)*np.random.rand(M, N)
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
      Np = 16 # num pulses
      Ntx = 4
      Nrx = 4
      
      M = Np*Ntx*Nrx
      N = np.round(np.sqrt(2*M))
      
      m = np.arange(M)[:,None]
      n = np.arange(N)

      v = np.cos(np.pi*np.outer(np.arange(Np),n)/N)
      hrx = np.cos(np.pi*np.outer(np.arange(Nrx),n)/N)
      htx = np.cos(np.pi*np.outer(np.arange(Ntx),n)/N)
      h = khatri_rao(hrx,htx)
      self.A = np.kron(h,v)

    # normalize cols
    # self.A = np.matmul(self.A,np.diag(1/np.sqrt(np.sum(self.A**2,axis=0))))
    return

  
    
  def size(self, dim):
    return self.A.shape[dim]

def khatri_rao(a, b):
    c = a[...,:,np.newaxis,:] * b[...,np.newaxis,:,:]
    # collapse the first two axes
    return c.reshape((-1,) + c.shape[2:])

#L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
# class GaussianProblem(Problem):