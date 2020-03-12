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
    if p.scen == 'siso':
      self.params_init = {'lambda':0.02 , 'alpha':1. , 'rho':1.}
    elif p.scen == 'siso_d':
      self.params_init = {'lambda':0.1 , 'alpha':1. , 'rho':1.}
    elif p.scen == 'mimo':
      self.params_init = {'lambda':0.02 , 'alpha':1. , 'rho':1.}
    elif p.scen == 'gaussian':
      self.params_init = {'lambda':0.01 , 'alpha':1. , 'rho':1.}
    
    if 'params_init' in kwargs.keys():
      for k,v in kwargs['params_init'].items():
        self.params_init[k] = v
    
    if 'tied' in args:
      self.tied = 'tied'
    elif 'untied' in args:
      self.tied = 'untied'

    self.n1 = p.size(1)
    self.Layers=[]
    for i in range(num_stages):
      self.Layers.append(Stage(self.params_init, p, *args))
    
    if 'blank' in args:
      print('BLANK ADMM-Net with {0} stages'.format(len(self.Layers)))
    elif 'tied' in args:
      print('TIED ADMM-Net with {0} stages and initial parameters:'.format(len(self.Layers)))
    else:
      print('UNTIED ADMM-Net with {0} stages and initial parameters:'.format(len(self.Layers)))
    print('Scenario:','{0}x{1}'.format(p.size(0),p.size(1)),p.scen)
    
    for k,v in self.params_init.items():
      print(k,'=',v)

  def call(self, inputs):
    z_0 = np.zeros((self.n1,1),dtype=np.float32)
    u_0 = np.zeros((self.n1,1),dtype=np.float32)
    z,u = self.Layers[0](inputs, z_0, u_0)
    for l in self.Layers[1:]:
      z,u = l(inputs,z,u)
    return tf.transpose(z)

class Stage(layers.Layer):

  def __init__(self, params_init, p, *args):
    super().__init__()
    self.x_update = x_update(params_init, p, *args)
    self.z_update = z_update(params_init, p, *args)
    # self.u_update = u_update(*args, **kwargs)
  
  def call(self, inputs, z, u):
    x_1 = self.x_update(inputs,z,u)
    z_1 = self.z_update(x_1,u)
    u_1 = u + x_1 - z_1
    # u_1 = self.u_update(x_1,z_1,u)
    return z_1,u_1
    
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
      AULA = self.AULA(p)
      M1_init = (1/self.rho0)*p.A.T.conj() - (1/self.rho0**2)*np.matmul(AULA,p.A.T.conj())
      # M2_init = np.eye(n) - (1/self.rho0)*AULA
      M2_init = (1/self.rho0)*(np.eye(n) - (1/self.rho0)*AULA)
      M3_init = np.ones((n,))

    if 'tied' in args:
      tied = True
    else:
      tied = False

    # self.M1 = tf.Variable(initial_value=M1_init.astype(np.float32),
    #                      trainable=not tied, name='M1')
    self.M2 = tf.Variable(initial_value=M2_init.astype(np.float32),
                         trainable=not tied, name='M2')
    self.M3 = tf.Variable(initial_value=M3_init.astype(np.float32),
                         trainable=True, name='M3')

    self.alph = tf.Variable(initial_value=self.alpha0,
                         trainable=True, name='alpha')
    self.rho = tf.Variable(initial_value=self.rho0,
                         trainable=False, name='rho')

  def call(self, y, z, u):
    # x = tf.matmul(self.M1, tf.transpose(y)) + tf.matmul(self.M2, z-u)
    n = z.shape[0]
    x = tf.matmul(tf.linalg.diag(self.M3),tf.matmul(self.M2, tf.transpose(y)) + tf.matmul(self.M2, self.rho*(z-u)))
    return self.alph*x + (1-self.alph)*z
    
  def AULA(self,p):
    M = p.size(0)
    N = p.size(1)
    L = np.linalg.cholesky(np.eye(M) + (1/self.rho0)*np.matmul(p.A,p.A.T.conj()))
    U = L.T.conj();
    #AULA = (A'*(U \ ( L \ A )));
    return np.matmul(p.A.T.conj(),np.matmul(la.inv(U),np.matmul(la.inv(L),p.A)))

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