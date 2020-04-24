#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:40:31 2020

@author: jeremyjohnston
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import numpy.linalg as la

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp

class MeanPercentageSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      err_norm = tf.math.reduce_sum(math_ops.square(y_pred - y_true),axis=-1)
      y_true_norm = tf.math.reduce_sum(math_ops.square(y_true),axis=-1)
      return K.mean(tf.math.divide(err_norm,y_true_norm))

class ADMMNet(tf.keras.Model):
  
  def __init__(self, p, num_stages, *args, **kwargs):
    super().__init__()
    # p = kwargs['problem']
    # if p.scen == 'siso':
    #   self.params_init = {'lambda':0.02 , 'alpha':1. , 'rho':1.}
    # elif p.scen == 'siso_d':
    #   self.params_init = {'lambda':0.1 , 'alpha':1. , 'rho':1.}
    # elif p.scen == 'mimo':
    #   self.params_init = {'lambda':0.02 , 'alpha':1. , 'rho':1.}
    # elif p.scen == 'mimo_c':
    #   self.params_init = {'lambda':1e-4 , 'alpha':4.64e-1 , 'rho':1e-4}
    # elif p.scen == 'gaussian':
    #   self.params_init = {'lambda':0.01 , 'alpha':1. , 'rho':1.}
    # else:
    self.params_init = {'lambda':0.1,'lambda2':0.1 , 'alpha':1. , 'rho':1.}
    
    if 'params_init' in kwargs.keys():
      for k,v in kwargs['params_init'].items():
        self.params_init[k] = v
    
    if 'tied' in args:
      self.tied = 'tied'
    elif 'untied' in args:
      self.tied = 'untied'

    self.n1 = p.A.shape[1]
    self.Layers=[]
    for i in range(num_stages):
      self.Layers.append(Stage(self.params_init, p, *args))
    
    if 'blank' in args:
      print('BLANK ADMM-Net with {0} stages'.format(len(self.Layers)))
    elif 'tied' in args:
      print('TIED ADMM-Net with {0} stages and initial parameters:'.format(len(self.Layers)))
    else:
      print('UNTIED ADMM-Net with {0} stages and initial parameters:'.format(len(self.Layers)))
    
    if p.partition == True:
      print('Scenario:','{0}x{1}'.format(p.size(0),p.size(1)),p.scen,'partition')
    else:
      print('Scenario:','{0}x{1}'.format(p.size(0),p.size(1)),p.scen)
    
    for k,v in self.params_init.items():
      print(k,'=',v)

  def call(self, inputs):
    z_0 = np.zeros((2*self.n1,1),dtype=np.float32)
    u_0 = np.zeros((2*self.n1,1),dtype=np.float32)
    z,u = self.Layers[0](inputs, z_0, u_0)
    for l in self.Layers[1:]:
      z,u = l(inputs,z,u)
    return tf.transpose(z)

class Stage(layers.Layer):

  def __init__(self, params_init, p, *args):
    super().__init__()
    m = p.size(0);
    n = p.size(1);
    self.m = m
    self.n = n
    self.p = p
    self.rho0 = params_init['rho']
    self.alpha0 = params_init['alpha']
    self.lambda0 = params_init['lambda']
    self.lambda0_2 = params_init['lambda2']
      
    if 'random_init' in args:
      M1_init = np.random.normal(size=(2*n,2*m))
      M2_init = np.random.normal(size=(2*n,2*n))
      # normalize cols
      M1_init = np.matmul(M1_init,np.diag(1/np.sqrt(np.sum(M1_init**2,axis=0))))
      M2_init = np.matmul(M2_init,np.diag(1/np.sqrt(np.sum(M2_init**2,axis=0))))
    else:
      AULA = self.AULA(p)
      M1 = np.matmul(np.eye(n)/self.rho0 - (1/self.rho0**2)*AULA, p.A.T.conj())
      M2 = np.eye(n) - (1/self.rho0)*AULA

      top = np.concatenate((M1.real, -M1.imag),axis=1)
      bot = np.concatenate((M1.imag, M1.real),axis=1)
      M1_init = np.concatenate((top,bot),axis=0)

      top = np.concatenate((M2.real, -M2.imag),axis=1)
      bot = np.concatenate((M2.imag, M2.real),axis=1)
      M2_init = np.concatenate((top,bot),axis=0)

    if 'tied' in args:
      tied = True
    else:
      tied = False

    self.M1 = tf.Variable(initial_value=M1_init.astype(np.float32),
                         trainable=not tied, name='M1')
    self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
                         trainable=not tied, name='M2')

    self.alph = tf.Variable(initial_value=self.alpha0,
                         trainable=True, name='alpha')

    self.beta = tf.Variable(initial_value=1.,
                         trainable=True, name='beta')

    self.rho = tf.Variable(initial_value=self.rho0,
                         trainable=False, name='rho')

    self.lamb = tf.Variable(initial_value=params_init['lambda'],
                            trainable=True, name='lambda')

    if self.p.partition == True:
      self.lamb2 = tf.Variable(initial_value=params_init['lambda2'],
                              trainable=True, name='lambda2')
                            
  
  def AULA(self,p):
    M = p.size(0)
    N = p.size(1)
    L = np.linalg.cholesky(np.eye(M) + (1/self.rho0)*np.matmul(p.A,p.A.T.conj()))
    U = L.T.conj();
    #AULA = (A'*(U \ ( L \ A )));
    return np.matmul(p.A.T.conj(),np.matmul(la.inv(U),np.matmul(la.inv(L),p.A)))

  def call(self, y, z, u):
    # x = tf.matmul(self.M1, tf.transpose(y)) + tf.matmul(self.M2, self.rho*(z-u))
    x = tf.matmul(self.M1, tf.transpose(y)) + tf.matmul(self.M2, (z-u))

    
    x = self.alph*x + (1-self.alph)*z

    
    v = self.re2comp(x+u)

    if self.p.partition == True:
      zc = self.z_update_partition(v)
    else:
      zc = self.z_update_no_partition(v)

    z = self.comp2re(zc)

    u = u + self.beta*(x - z)
    # u = u + x - z

    return z,u
  
  def z_update_partition(self, v):
    z1 = self.soft_thresh(v[:self.p.N_part], self.lamb/self.rho)
    z2 = self.soft_thresh(v[self.p.N_part:], self.lamb2/self.rho)
    return tf.concat((z1,z2),axis=0)

  def z_update_no_partition(self, v):
    return self.soft_thresh(v, self.lamb/self.rho)

  def comp2re(self, x):
    return tf.concat((x[:,0],x[:,1]), axis=0)

  def re2comp(self, x):
    # input: x is a shape (2N, 1) real-valued concatenation of a length-N complex vector
    # output: a shape (N, 2) array corresponding to the recomposed complex 
    
    ndiv2 = 2*self.n//2

    x_re = x[:ndiv2]
    x_im = x[ndiv2:]
    
    return tf.concat((x_re[:,None],x_im[:,None]), axis=1)

  def soft_thresh(self, x, kappa):
    # x is a shape (N,2) array whose rows correspond to complex numbers
    # returns shape (N,2) array corresponding to complex numbers

    # ndiv2 = 2*self.n//2

    # x1 = x[:ndiv2]
    # x2 = x[ndiv2:]
    # xx = tf.concat((x1[:,None],x2[:,None]), axis=1)
    
    x1 = x[:,0]
    x2 = x[:,1]

    norm = tf.norm(x,axis=1)
    x1normalized = tf.math.divide_no_nan(x1,norm)
    x2normalized = tf.math.divide_no_nan(x2,norm)

    z1 = tf.math.multiply(x1normalized,tf.maximum(norm - kappa,0))
    z2 = tf.math.multiply(x2normalized,tf.maximum(norm - kappa,0))

    return tf.concat((z1[:,None],z2[:,None]),axis=1)

class x_update(layers.Layer):

  def __init__(self, params_init, p, *args, **kwargs):
    super().__init__()
    # p = kwargs['problem']
    m = p.size(0);
    n = p.size(1);
    self.rho0 = params_init['rho']
    self.alpha0 = params_init['alpha']
    self.lambda0 = params_init['lambda']
      
    if 'blank' in args:
      M1_init = np.zeros((n,m))  
      M2_init = np.zeros((n,n))
      M3_init = np.zeros((n,))
    else:
      if m < n:
        AULA = self.AULA(p)
        M1_init = np.matmul(np.eye(n)/self.rho0 - (1/self.rho0**2)*AULA, p.A.T.conj())
        M2_init = (1/self.rho0)*np.eye(n) - (1/self.rho0**2)*AULA
      else:
        # UL = self.invUinvL(p)
        # M1_init = np.matmul(UL, p.A.T.conj())
        # M2_init = self.rho0*UL
        M1_init = np.random.normal(size=(n,m))
        M2_init = np.random.normal(size=(n,n))
        # M1_init = la.inv( np.matmul(p.A.T.conj(),p.A) + self.rho0*np.eye(n) ) * p.A.T.conj() 
        # M2_init = la.inv( np.matmul(p.A.T.conj(),p.A) + self.rho0*np.eye(n) ) * self.rho0
        # q = Atb + rho*(z - u); 
        # x = U \ (L \ q);
      # M3_init = np.ones((n,))

    if 'tied' in args:
      tied = True
    else:
      tied = False

    self.M1 = tf.Variable(initial_value=M1_init.astype(np.float32),
                         trainable=not tied, name='M1')
    self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
                         trainable=not tied, name='M2')
    # self.M3 = tf.Variable(initial_value=M3_init.astype(np.float32),
    #                      trainable=not tied, name='M3')

    self.alph = tf.Variable(initial_value=self.alpha0,
                         trainable=True, name='alpha')
    self.rho = tf.Variable(initial_value=self.rho0,
                         trainable=False, name='rho')

  def call(self, y, z, u):
    x1 = tf.matmul(self.M1, tf.transpose(y))
    x2 = tf.matmul(self.M2, self.rho*(z-u))
    x = x1 + x2
    # x = tf.matmul(self.M1, tf.transpose(y)) + tf.matmul(self.M2, self.rho*(z-u))
    # n = z.shape[0]
    # x = tf.matmul(tf.linalg.diag(self.M3),tf.matmul(self.M2, tf.transpose(y)) + tf.matmul(self.M2, self.rho*(z-u)))
    return self.alph*x + (1-self.alph)*z
    
  def AULA(self,p):
    M = p.size(0)
    N = p.size(1)
    L = np.linalg.cholesky(np.eye(M) + (1/self.rho0)*np.matmul(p.A,p.A.T.conj()))
    U = L.T.conj();
    #AULA = (A'*(U \ ( L \ A )));
    return np.matmul(p.A.T.conj(),np.matmul(la.inv(U),np.matmul(la.inv(L),p.A)))

  def invUinvL(self,p):
    M = p.size(0)
    N = p.size(1)
    L = np.linalg.cholesky( np.matmul(p.A.T.conj(),p.A) + self.rho0*np.eye(N))
    U = L.T.conj()
    return np.matmul(la.inv(U), la.inv(L))

class z_update(layers.Layer):

  def __init__(self, params_init, p, *args, **kwargs):
    super().__init__()
    # p = kwargs['problem']

    self.lamb = tf.Variable(initial_value=params_init['lambda'],
                            trainable=True, name='lambda')
    self.rho = tf.Variable(initial_value=params_init['rho'],
                            trainable=False, name='rho')
                            
  def call(self, x_1, u):
    return tfp.math.soft_threshold(x_1+u, self.lamb/self.rho)
    # return tf.keras.activations.relu(x_1 + u, alpha=self.lamb/self.rho, max_value=None, threshold=0) - tf.keras.activations.relu(x_1 + u, alpha=-self.lamb/self.rho, max_value=None, threshold=0)
    # return tf.sign(x_1 + u) * tf.maximum(tf.abs(x_1 + u) - self.lamb/self.rho, 0)

class u_update(layers.Layer):

  def __init__(self, *args, **kwargs):
    super().__init__()
    
  def call(self, x_1, z_1, u):
    return u + x_1 - z_1




    
    
# x = tf.ones((2, 2))
# linear_layer = Linear(4, 2)
# y = linear_layer(x)
# print(y)