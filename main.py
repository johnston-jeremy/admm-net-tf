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
import train_net as tn
import load_net as ln

def generate_data(A, sparsity, bs, SNR=np.inf, mismatch=False):
    m, n = A.shape
    x_real = np.zeros([n, bs])
    x_imag = np.zeros([n, bs])
    x = np.zeros([n, bs]) + 1j * np.zeros([n, bs])
    for i in range(bs):
        idx = rd.sample(list(range(n)), sparsity)
        # x[idx,i] = 1.*np.exp(1j*2*np.pi*np.random.rand(sparsity))
        # x[idx,i] = np.random.rand(sparsity) + 1j*np.random.rand(sparsity)
        x[idx,i] = 2 * ( np.random.rand(sparsity) + 1j*np.random.rand(sparsity) ) - 1 - 1j
        # x[idx,i] = 1 + 1j

    X = x.copy()
    Y = np.matmul(A,X)

    if mismatch == True:
      phi_e = 0.1*(np.random.rand(m,bs)-0.5)
      e = np.exp(1j*2*np.pi*phi_e)
      Y = Y*e
    # ----- Generate the compressed noiseless signals --------
      # Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
      # for i in range(bs):
      #     tmp = np.matmul(A, X[:, i])
      #     # tmp /= np.linalg.norm(tmp)
      #     Y[:, i] = tmp
    
    # ----- Generate the compressed signals with noise -------
    noise_std = np.power(10, -(SNR / 20))
    for i in range(bs):
        # tmp /= np.linalg.norm(tmp)
        Y[:, i] = Y[:, i] + noise_std * (np.random.randn(m) + 1j * np.random.randn(m))
    return Y.T, X.T

def generate_data_with_partition(A, n1, sparsity1, sparsity2, Ns, SNR=20, noise=False, mismatch=True):
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

  if mismatch == True:
      phi_e = 0.1*(np.random.rand(m,bs)-0.5)
      e = np.exp(1j*2*np.pi*phi_e)
      Y = Y*e
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

def print_best(errs, names, L, params):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params

  print('lambda1', 'lambda2', 'rho','alpha','err',sep='\t\t')
  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3): 
        for l4 in range(L4):     
          print('{:.2e}'.format(lam[l3]), '{:.2e}'.format(lam2[l4]), '{:.2e}'.format(rho[l1]), '{:.2e}'.format(alph[l2]), round(errs[l1,l2,l3,l4],2), sep='\t')     
  l1min,l2min,l3min,l4min = np.unravel_index(np.argmin(errs),errs.shape)
  print('lambda1', '{:.2e}'.format(lam[l3min]) )
  print('lambda2', '{:.2e}'.format(lam2[l4min]) )
  print('alpha', '{:.2e}'.format(alph[l2min]) )
  print('rho', '{:.2e}'.format(rho[l1min]) )
  # print(names[l1min,l2min,l3min])
  print('Error:', round(errs[l1min,l2min,l3min,l4min],2), 'dB')
  print('File:', names[l1min,l2min,l3min,l4min])

def eval_nets(nets, data_test, L, params_initialization, names):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params_initialization

  errs = np.zeros((L1,L2,L3,L4))

  x_test,y_test = data_test
  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3):
        for l4 in range(L4):
          xhat = nets[l1,l2,l3,l4].predict_on_batch(y_test)
          errs[l1,l2,l3,l4] = 10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1))))
          # names[l1,l2,l3,l4] = tn.save_net(a,props)
  
  print_best(errs, 
             names, 
             L, 
             params_initialization
             )
  return errs
      
def train_various_param_inits(props, L, params_initialization, data_train,data_test):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params_initialization

  names = np.empty((L1,L2,L3,L4),dtype=object)
  nets = np.empty((L1,L2,L3,L4),dtype=object)
  
  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3):
        for l4 in range(L4):
          params_init = {'lambda':lam[l3] , 'lambda2':lam2[l4], 'alpha':alph[l2] , 'rho':rho[l1], 'beta':1.}
          a = tn.gen_net(props, 
                      'untied',
                      params_init=params_init
              )
          
          a = tn.train_net(a, data_train, props)
          names[l1,l2,l3,l4] = tn.save_net(a,props)
          nets[l1,l2,l3,l4] = a
             
  return nets, names

def load_net(folder, scen):
  # args = ln.get_args()
  # scen = args[0].scen
  # folder = './nets/' + args[0].folder

  A = np.load(folder + '/A.npy')
  n1,n2 = A.shape
  p = problem.Problem((n1,n2-n1),scen,partition=True, N_part=n2-n1)
  
  a = admm.ADMMNet(p,10,'untied')
  # a = lista.ISTANet('blank', problem=p)
  a.load_weights(folder + '/weights')
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  a.compile(optimizer, loss=admm.MeanPercentageSquaredError())
  return a

def train_old():
  a = load_net()
  
  errs = np.zeros((L1,L2,L3,L4))
  names = np.zeros((L1,L2,L3,L4),dtype=object)

  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3):
        for l4 in range(L4):
          a = tn.train_net(a, p, (x,y), props, epochs=epochs, batch_size=batch_size)
          xhat = a.predict_on_batch(y_test)
          errs[l1,l2,l3,l4] = 10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1))))
          names[l1,l2,l3,l4] = tn.save_net(a,props)

  return errs, names

def problem_setup(SNR=np.inf):
  m=8
  n=16
  p = problem.Problem((m,n), 'siso', partition=False, N_part=n)
  m,n = p.A.shape
  n = n - m
  # x,y = gen_samples(p.A, 1e5, density=0.2, normalize=False, ones=False, comp=True)

  train_number = int(1e5)
  test_number = int(1e5)
  sparsity = 2
  sparsity1 = 2
  sparsity2 = 1

  y,x = generate_data(p.A, sparsity, train_number, SNR=SNR, mismatch=False)
  # y,x = generate_data_with_partition(p.A, n, sparsity1, sparsity2, train_number, SNR=SNR, noise=True)
  y = np.concatenate((y.real,y.imag),axis=1)
  x = np.concatenate((x.real,x.imag),axis=1)

  y_test,x_test = generate_data(p.A, sparsity, train_number, SNR=SNR, mismatch=True)
  # y_test,x_test = generate_data_with_partition(p.A, n, sparsity1, sparsity2, test_number, SNR=SNR, noise=True)
  y_test = np.concatenate((y_test.real,y_test.imag),axis=1)
  x_test = np.concatenate((x_test.real,x_test.imag),axis=1)
  # np.save('/Users/jeremyjohnston/Documents/admm-net-tf/trainingdata/1e6_mimo_d_part_8x26_x',x)
  # np.save('/Users/jeremyjohnston/Documents/admm-net-tf/trainingdata/1e6_mimo_d_part_8x26_y',y)

  # x = np.load('/Users/jeremyjohnston/Documents/admm-net-tf/trainingdata/1e6_mimo_d_part_8x26_x.npy')
  # y = np.load('/Users/jeremyjohnston/Documents/admm-net-tf/trainingdata/1e6_mimo_d_part_8x26_y.npy')
  # x = x[:train_number]
  # y = y[:train_number]
  data_train = x,y
  data_test = x_test,y_test
  return p, data_train, data_test

SNR=np.inf
p, data_train, data_test = problem_setup(SNR=SNR)

props = dict(net_type='admm', 
          num_stages=10, 
          problem=p,
          learning_rate=1e-3,
          optimizer='adam',
          schedule=False,
          Ntrain=data_train[0].shape[0], #num training samples
          batch_size=2**10, # batch size
          epochs=1,
          SNR=SNR
      )


# lambda alpha rho
# L3 = 1; L2 = 1; L1 = 1
L4 = 1; L3 = 1; L2 = 1; L1 = 1
# lam = [2e-2]; alph=[1.8]; rho = [1.] # 1 + 1j only
lam = [1e-2]; lam2 = [4e-3]; alph=[1.8e-0]; rho = [1e-0]
# lam = np.logspace(-2,-1,L3, dtype=np.float32)
# lam2 = np.logspace(-4,-3,L3, dtype=np.float32)

# alph = 1*np.logspace(-2,-0.,L2, dtype=np.float32)
# alph = np.linspace(1,1.9,L2,dtype=np.float32)

# rho = np.logspace(-1,0,L1, dtype=np.float32)
# rho = np.linspace(1,1.9,L1,dtype=np.float32)

L = (L1,L2,L3,L4)
params_initialization = (lam,lam2,alph,rho)

nets, names = train_various_param_inits(props, L, params_initialization, data_train, data_test)
errs = eval_nets(nets, data_test, L, params_initialization, names)

# trained_nets = train_new(props, data_train, data_test)

# errs,names = train_old()

# loadnet()

# x,y = gen_samples(p.A,1e4,density=d, normalize=False, ones=False)
# xhat = a.predict_on_batch(y)
# print(10*np.log10(np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1)))))