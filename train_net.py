import problem
import admm
import admm_interference as admm_i
import lista
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def gen_samples(A,N,density=0.1,normalize=False,comp=False, ones=True):
  N = int(N)
  n1,n2 = A.shape

  # x = sp.sparse.random(n2,N,density,dtype=np.float32).A
  # y = np.matmul(A,x)

  # x = x.T
  # y = y.T

  x = np.zeros((N,n2))
  for i in range(N):
    x[i] = sp.sparse.random(1,n2,density,dtype=np.float32,format='array')[0]
    # y1[i] = np.matmul(A,x1[i])
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

def gen_net(properties, adamrate, *args, **kwargs):
  f = properties['net_type']
  n = properties['num_stages']
  p = properties['problem']
  
  if f == 'lista':
    a = lista.ISTANet(p, num_stages=n, params_init=opts['params_init'], *args )
  elif f == 'admm':
    a = admm.ADMMNet(p, n, *args, **kwargs)
  elif f == 'admm_i':
    a = admm_i.ADMMNet(p, n, *args, **kwargs)
  
  if 'schedule' in args:
    STEPS_PER_EPOCH = N//1000
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=STEPS_PER_EPOCH*1000,
      decay_rate=1,
      staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=adamrate)

  a.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

  return a

def train_net(a, p, data, properties, **kwargs):
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

  labels,y = data
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
  params_init = {'lambda':4e-5 , 'alpha':1.9 , 'rho':1e-2}
  # params_init = {'lambda':[], 'alpha':1.5, 'rho':1.5}

  p = problem.Problem((16,32), 'mimo')
  # d = scipy.io.loadmat('matvars.mat')
  # p.A = d['A']

  # sparsity density
  d = 0.05

  # x,y = gen_samples(p.A, 5e5, density=d, normalize=False, ones=False)
  # np.save('./trainingdata/5e5_mimo_16x32_x',x)
  # np.save('./trainingdata/5e5_mimo_16x32_y',y)

  # y = np.matmul(p.A.T.conj(),y.T).T

  x = np.load('./trainingdata/5e5_mimo_16x32_x.npy')
  y = np.load('./trainingdata/5e5_mimo_16x32_y.npy')
  
  props = dict(net_type='admm', 
              num_stages=6, 
              problem=p
          )

  a = gen_net(props, 
              1e-4, 
              'tied', 
              params_init=params_init
      )

  a = train_net(a, p, (x,y), props, epochs=200)

  x,y = gen_samples(p.A,1e4,density=d, normalize=False, ones=False)

  # xhat = a.predict_on_batch(np.matmul(y,p.A))
  xhat = a.predict_on_batch(y)
  
  # xhat = a.predict(d['y'].T)
  # x = (d['x'].A).T
  # print_vars(a)
  # xhat = a.predict(y,batch_size=1000)
  
  print(10*np.log10(np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1)))))