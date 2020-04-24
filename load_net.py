import problem
import admm2 as admm
import lista
import tensorflow as tf
import numpy as np
import scipy as sp
import time
import numpy.linalg as la
import argparse
import os
import random as rd
# import cvx_test


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def generate_data_with_partition(A, n1, sparsity1, sparsity2, Ns, SNR=20, noise=False):
  m, n = A.shape
  n2 = n - n1

  x1 = np.zeros((n1, Ns),dtype=complex)
  x2 = np.zeros((n2, Ns),dtype=complex)
  for i in range(Ns):
      idx1 = rd.sample(list(range(n1)), sparsity1)
      idx2 = rd.sample(list(range(n2)), sparsity2)
      # x[idx,i] = 1.*np.exp(1j*2*np.pi*np.random.rand(sparsity))
      # x[idx,i] = np.random.rand(sparsity) + 1j*np.random.rand(sparsity)
      x1[idx1,i] = 2 * ( np.random.rand(sparsity1) + 1j*np.random.rand(sparsity1) ) - 1 - 1j
      x2[idx2,i] = 2 * ( np.random.rand(sparsity2) + 1j*np.random.rand(sparsity2) ) - 1 - 1j
      # x[idx,i] = 1 + 1j

  X = np.concatenate((x1,x2),axis=0)
  # X = x.copy()
  Y = np.matmul(A,X)
  # ----- Generate the compressed noiseless signals --------
    # Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    # for i in range(bs):
    #     tmp = np.matmul(A, X[:, i])
    #     # tmp /= np.linalg.norm(tmp)
    #     Y[:, i] = tmp
  
  # ----- Generate the compressed signals with noise -------
  if noise == True:
    noise_std = np.power(10, -(SNR / 20))
    for i in range(Ns):
        # tmp /= np.linalg.norm(tmp)
        Y[:, i] = Y[:, i] + noise_std * (np.random.randn(m) + 1j * np.random.randn(m))
  return Y.T, X.T

def get_args():
  parser = argparse.ArgumentParser()
  net_arg = parser.add_argument_group ('net')
  net_arg.add_argument (
    '-f', '--folder', type=str,
    help='Folder name.')
  net_arg.add_argument (
    '-s', '--scen', type=str,
    help='Scenario.')
  return parser.parse_known_args()

def get_net(num_stages, *args, **kwargs):
  cl_args = get_args()
  scen = cl_args[0].scen
  folder = './nets/' + cl_args[0].folder #03_05_2020_04_05_38'
  
  A = np.load(folder + '/A.npy')
  n1,n2 = A.shape
  p = problem.Problem((n1,n2),scen)

  net = admm.ADMMNet(p,num_stages,*args,**kwargs)
  # a = lista.ISTANet('blank', problem=p)
  net.load_weights(folder + '/weights')
  
  return net

if __name__ == "__main__":
  args = get_args()
  scen = args[0].scen
  folder = './nets/' + args[0].folder #03_05_2020_04_05_38'

  A = np.load(folder + '/A.npy')
  n1,n2 = A.shape
  p = problem.Problem((n1,n2-n1),scen,partition=True, N_part=n2-n1)
  
  a = admm.ADMMNet(p,10)
  # a = lista.ISTANet('blank', problem=p)
  a.load_weights(folder + '/weights')
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  a.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

  # for i in range(10):
  #   printvars = ['lambda:0','alpha:0','rho:0']
  #   print('Layer', i)
  #   for v in a.layers[i].variables:
  #     if v.name in printvars:
  #       print(v.name + ' =', v.numpy())

  

  # m=8
  # n=16
  # p = problem.Problem((m,n), 'mimo_d', partition=True, N_part=n)
  m,n = p.A.shape
  n = n - m
  # x,y = gen_samples(p.A, 1e5, density=0.2, normalize=False, ones=False, comp=True)

  test_number = int(1e5)
  sparsity1 = 2
  sparsity2 = 1
  SNR = 40

  y_test,x_test = generate_data_with_partition(p.A, n, sparsity1, sparsity2, test_number, SNR=SNR, noise=True)
  y_test = np.concatenate((y_test.real,y_test.imag),axis=1)
  x_test = np.concatenate((x_test.real,x_test.imag),axis=1)

  # ADMM-Net
  xhat = a.predict_on_batch(y_test)
  print(10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1)))))

  # CVX
  # xhat_cvx = cvx_test.cvx_test_mp(p,y)
  # print(np.mean((np.sum((x-xhat_cvx)**2,axis=1)/np.sum(x**2,axis=1))))