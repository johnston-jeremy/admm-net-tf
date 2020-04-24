import problem
import admm2 as admm
import admm_interference as admm_i
import lista
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import time
import load_net
import random as rd
from gen_samples import gen_samples

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def gen_net(properties, *args, **kwargs):
  f = properties['net_type']
  n = properties['num_stages']
  p = properties['problem']
  rate = properties['learning_rate']
  optim = properties['optimizer']
  schedule = properties['schedule']
  Ntrain = properties['Ntrain']
  batch_size = properties['batch_size']
  
  if f == 'lista':
    a = lista.ISTANet(p, n, *args, **kwargs)
  elif f == 'admm':
    a = admm.ADMMNet(p, n, *args, **kwargs)
  elif f == 'admm_i':
    a = admm_i.ADMMNet(p, n, *args, **kwargs)
  
  if schedule == True:
    STEPS_PER_EPOCH = Ntrain//batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      rate,
      decay_steps=STEPS_PER_EPOCH*10,
      decay_rate= 1,
      staircase=True)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #   adamrate,
    #   decay_steps=STEPS_PER_EPOCH*100,
    #   decay_rate=.5,
    #   staircase=True)  
    # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #   boundaries=[500], values=list(rate*np.array([1e0,1e-1])), name=None
    # )
    # breakpoint()
    import matplotlib.pyplot as plt
    step = np.linspace(0,10000)
    lr = lr_schedule(step)
    plt.figure(figsize = (8,6))
    plt.plot(step/STEPS_PER_EPOCH, lr)
    plt.ylim([0,max(plt.ylim())])
    plt.xlabel('Epoch')
    _ = plt.ylabel('Learning Rate')
    plt.show()

    optimizer = tf.keras.optimizers.Adam(lr_schedule)
  else:
    if optim == 'adam':
      optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
    elif optim == 'sgd':
      optimizer = tf.keras.optimizers.SGD(learning_rate=rate, momentum=0.9)


  # a.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
  a.compile(optimizer, loss=admm.MeanPercentageSquaredError())

  return a

def train_net(a, data, props, **kwargs):

  a.fit(data[1], 
        data[0], 
        epochs=props['epochs'], 
        batch_size = props['batch_size']
  )
  
  return a

def save_net(net, properties):
  f = properties['net_type']
  n = properties['num_stages']
  p = properties['problem']
  snr = properties['SNR']

  filename = f + str(n)  + net.tied + '_' + p.scen + '_{0}x{1}'.format(p.size(0),p.size(1)) + '_' + 'SNR' + str(snr) + 'dB_' + time.strftime("%m_%d_%Y_%H_%M_%S",time.localtime())
  filepath = './nets/' + filename
  os.mkdir(filepath)
  net.save_weights(filepath+'/weights')
  np.save(filepath+'/A', p.A)

  return filename

def test_net(a, N=1e3):
  x,y = gen_samples(p.A,N)
  xhat = a.predict_on_batch(y)
  print(10*np.log10(np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1)))))
  return

def loadnet():
  p = problem.Problem((20,40), 'mimo')


  # x = np.load('./trainingdata/1e5_mimo_c_4x8_x.npy')
  # y = np.load('./trainingdata/1e5_mimo_c_4x8_y.npy')
  
  x = np.load('./trainingdata/1e6_mimo_20x40_x.npy')
  y = np.load('./trainingdata/1e6_mimo_20x40_y.npy')

  props = dict(net_type='admm', 
              num_stages=4, 
              problem=p
          )
          
  a = load_net.get_net(4, 'untied', params_init=params_init)

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
  # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=False)

  # a.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
  a.compile(optimizer, loss=lista.MeanPercentageSquaredError())

  a = train_net(a, p, (x,y), props, epochs=100)

  save_net(a,props)

  # x = np.load('./trainingdata/1e5_mimo_20x40_x.npy')
  # y = np.load('./trainingdata/1e5_mimo_20x40_y.npy')

  xhat = a.predict_on_batch(y)
  breakpoint()
  print(10*np.log10(np.mean(np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1))))

def print_vars(a):
  printvars = ['lambda:0','alpha:0']
  for l in a.Layers:
    for v in l.variables:
      if v.name in printvars:
        print(v.name + ' =', v.numpy())