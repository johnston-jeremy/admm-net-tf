import problem
import admm 
import lista
import tensorflow as tf
import numpy as np
import scipy as sp
import time
import numpy.linalg as la
import argparse
import os
# import cvx_test
from train_net import gen_samples

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


if __name__ == "__main__":
  args = get_args()
  scen = args[0].scen
  folder = './nets/' + args[0].folder #03_05_2020_04_05_38'
  
  A = np.load(folder + '/A.npy')
  n1,n2 = A.shape
  p = problem.Problem((n1,n2),scen)

  a = admm.ADMMNet(p,20,'blank')
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

  N = 1000
  x,y = gen_samples(p.A,1e4,density=0.01, normalize=False)

  # ADMM-Net
  xhat = a.predict_on_batch(np.matmul(y,A))
  print(10*np.log10(np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1)))))

  # CVX
  # xhat_cvx = cvx_test.cvx_test_mp(p,y)
  # print(np.mean((np.sum((x-xhat_cvx)**2,axis=1)/np.sum(x**2,axis=1))))