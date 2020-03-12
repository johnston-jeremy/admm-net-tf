import problem
import admm
import lista
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def gen_samples(A,N,density=0.1,normalize=True):
  N = int(N)
  n1,n2 = A.shape
  y = np.zeros((N,n1))
  x = np.zeros((N,n2))
  for i in range(N):
    x[i] = sp.sparse.random(n2,1,density,dtype=np.float32).A.T
    if normalize == True:
      x[i] = x[i]/np.sqrt(np.sum(x[i]**2))
    y[i] = np.matmul(A,x[i]).T

  return x,y

def gen_net(properties, *args, **kwargs):
  f = properties['net_type']
  n = properties['num_stages']
  p = properties['problem']
  
  if f == 'lista':
    a = lista.ISTANet(p, num_stages=n, params_init=opts['params_init'], *args )
  elif f == 'admm':
    a = admm.ADMMNet(p, n, *args, **kwargs)
  
  if 'schedule' in args:
    STEPS_PER_EPOCH = N//1000
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=STEPS_PER_EPOCH*1000,
      decay_rate=1,
      staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

  a.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

  return a

def train_net(a, p, properties, **kwargs):
  opts = dict(save=True,
                schedule=False,
                batch=1000,
                epochs=5)
  for k,v in kwargs.items():
    if k in opts.keys():
      opts[k] = v

  f = properties['net_type']
  n = properties['num_stages']
  p = properties['problem']

  a.fit(y, labels, epochs=opts['epochs'], batch_size = opts['batch'])

  if opts['save'] == True:
    filepath = './nets/' + f + str(n)  + a.tied + '_' + p.scen + '{0}x{1}'.format(p.size(0),p.size(1)) + '_' + time.strftime("%m_%d_%Y_%H_%M_%S",time.localtime())
    os.mkdir(filepath)
    a.save_weights(filepath+'/weights')
    np.save(filepath+'/A', p.A)
  
  return a

def test_net(a, N=1e3):
  x,y = gen_samples(p.A,N)
  xhat = a.predict_on_batch(y)
  print(10*np.log10(np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1)))))
  return

def print_vars(a):
  printvars = ['lambda:0','alpha:0']
  for l in a.Layers:
    for v in l.variables:
      if v.name in printvars:
        print(v.name + ' =', v.numpy())

if __name__ == '__main__':
  params_init = {'lambda':0.2, 'alpha':1.5, 'rho':1.5}

  p = problem.Problem((100,200), 'mimo')
  # d = scipy.io.loadmat('matvars.mat')
  # p.A = d['A']

  # sparsity density
  d = 0.01

  labels,y = gen_samples(p.A, 5e4,density=d, normalize=False)

  props = dict(net_type='admm', num_stages=20, problem=p)

  a = gen_net(props, 'tied', params_init=params_init)
  a = train_net(a, p, props, epochs=10)

  x,y = gen_samples(p.A,1e3,density=d, normalize=False)

  xhat = a.predict_on_batch(y)
  
  # xhat = a.predict(d['y'].T)
  # x = (d['x'].A).T
  # print_vars(a)
  # xhat = a.predict(y,batch_size=1000)
  
  print(10*np.log10(np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1)))))