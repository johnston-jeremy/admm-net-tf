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
import matplotlib.pyplot as plt

def generate_data(A, sparsity, Nsamp, SNR, mismatch):
    m, n = A.shape
    x_real = np.zeros([n, Nsamp])
    x_imag = np.zeros([n, Nsamp])
    x = np.zeros([n, Nsamp]) + 1j * np.zeros([n, Nsamp])
    for i in range(Nsamp):
        idx = rd.sample(list(range(n)), sparsity)
        # x[idx,i] = 1.*np.exp(1j*2*np.pi*np.random.rand(sparsity))
        # x[idx,i] = np.random.rand(sparsity) + 1j*np.random.rand(sparsity)
        x[idx,i] = 2 * ( np.random.rand(sparsity) + 1j*np.random.rand(sparsity) ) - 1 - 1j
        # x[idx,i] = 1 + 1j

    X = x.copy()
    Y = np.matmul(A,X)

    if mismatch == True:
      phi_e = 0.01*(2*np.random.rand(m,Nsamp) - 1)
      # phi_e = 0.1*(np.random.rand(m,Nsamp) - 0.5)
      e = np.exp(1j*2*np.pi*phi_e)
      Y = Y*e
    # ----- Generate the compressed noiseless signals --------
      # Y = np.zeros((m, Nsamp)) + 1j * np.zeros((m, Nsamp))
      # for i in range(Nsamp):
      #     tmp = np.matmul(A, X[:, i])
      #     # tmp /= np.linalg.norm(tmp)
      #     Y[:, i] = tmp
    
    # ----- Generate the compressed signals with noise -------
    noise_std = np.power(10, -(SNR / 20))
    for i in range(Nsamp):
        # tmp /= np.linalg.norm(tmp)
        Y[:, i] = Y[:, i] + noise_std * (np.random.randn(m) + 1j * np.random.randn(m))
    return Y.T, X.T

def generate_data_with_partition(A, n1, sparsity1, sparsity2, Nsamp, SNR, noise, mismatch):
  m, n = A.shape
  n2 = n - n1

  x1 = np.zeros((n1, Nsamp),dtype=complex)
  x2 = np.zeros((n2, Nsamp),dtype=complex)
  for i in range(Nsamp):
      idx1 = rd.sample(list(range(n1)), sparsity1)
      idx2 = rd.sample(list(range(n2)), sparsity2)
      # x[idx,i] = 1.*np.exp(1j*2*np.pi*np.random.rand(sparsity))
      # x[idx,i] = np.random.rand(sparsity) + 1j*np.random.rand(sparsity)
      x1[idx1,i] = 2 * ( np.random.rand(sparsity1) + 1j*np.random.rand(sparsity1) ) - 1 - 1j
      x2[idx2,i] = 0.3 * (2 * ( np.random.rand(sparsity2) + 1j*np.random.rand(sparsity2) ) - 1 - 1j)
      # x[idx,i] = 1 + 1j

  X = np.concatenate((x1,x2),axis=0)
  # X = x.copy()
  Y = np.matmul(A,X)

  if mismatch == True:
      phi_e = 0.1*(np.random.rand(m,Nsamp)-0.5)
      e = np.exp(1j*2*np.pi*phi_e)
      Y = Y*e
  
  # ----- Generate the compressed signals with noise -------
  if noise == True:
    nn = np.power(10, -(SNR / 20)) * (np.random.randn(m,Nsamp) + 1j*np.random.randn(m,Nsamp))
    Y = Y + nn
    # for i in range(Nsamp):
    #     # tmp /= np.linalg.norm(tmp)
    #     Y[:, i] = Y[:, i] + noise_std * (np.random.randn(m) + 1j * np.random.randn(m))
  return Y.T, X.T

def print_best(errs, names, L, params, quiet):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params
  l1min,l2min,l3min,l4min = np.unravel_index(np.argmin(errs),errs.shape)
  if quiet==False:
    # print the error for each parameter setting
    print('lambda1', 'lambda2', 'rho','alpha','err',sep='\t\t')
    for l1 in range(L1):
      for l2 in range(L2):
        for l3 in range(L3): 
          for l4 in range(L4):     
            print('{:.2e}'.format(lam[l3]), '{:.2e}'.format(lam2[l4]), '{:.2e}'.format(rho[l1]), '{:.2e}'.format(alph[l2]), round(errs[l1,l2,l3,l4],2), sep='\t')     
  
    print('lambda1', '{:.2e}'.format(lam[l3min]) )
    print('lambda2', '{:.2e}'.format(lam2[l4min]) )
    print('alpha', '{:.2e}'.format(alph[l2min]) )
    print('rho', '{:.2e}'.format(rho[l1min]) )

  # print(names[l1min,l2min,l3min])
  print('\nFile:', names[l1min,l2min,l3min,l4min])
  print('Error:', round(errs[l1min,l2min,l3min,l4min],2), 'dB')
  
def eval_nets(nets, data_test, L, params_initialization, names, quiet):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params_initialization
  
  errs = np.zeros((L1,L2,L3,L4))

  x_test,y_test = data_test
  N = y_test.shape[1]
  M = x_test.shape[1]
  Ng = M - N

  
  x_test = x_test[:,0:M//2] + 1j*x_test[:,M//2:]
  x1_test_c = x_test[:,0:Ng//2]
  x1_test_r = np.concatenate((x1_test_c.real,x1_test_c.imag),axis=1)

  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3):
        for l4 in range(L4):
          xhat = nets[l1,l2,l3,l4].predict_on_batch(y_test)
          xhat = xhat.numpy()
          xhat = xhat[:,0:M//2] + 1j*xhat[:,M//2:]
          x1_hat_c = xhat[:,0:Ng//2]
          x1_hat_r = np.concatenate((x1_hat_c.real,x1_hat_c.imag),axis=1)
          errs[l1,l2,l3,l4] = 10*np.log10(np.mean(np.sum((x1_test_r-x1_hat_r)**2,axis=1)/np.sum(x1_test_r**2,axis=1)))
          # errs[l1,l2,l3,l4] = 10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1))))
          # names[l1,l2,l3,l4] = tn.save_net(a,props)
  
  print_best(errs, 
             names, 
             L, 
             params_initialization,
             quiet
             )
  return errs
      
def eval_net(net, data_test, name, show_plots):

  x_test,y_test = data_test
  N = y_test.shape[1]
  M = x_test.shape[1]
  Ng = M - N

  xhat = net.predict_on_batch(y_test)
  xhat = xhat.numpy()
  xhat = xhat[:,0:M//2] + 1j*xhat[:,M//2:]
  x_test = x_test[:,0:M//2] + 1j*x_test[:,M//2:]

  x1_hat_c = xhat[:,0:Ng//2]
  x1_test_c = x_test[:,0:Ng//2]

  x1_hat_r = np.concatenate((x1_hat_c.real,x1_hat_c.imag),axis=1)
  x1_test_r = np.concatenate((x1_test_c.real,x1_test_c.imag),axis=1)

  # xhat = np.concatenate()
  err = 10*np.log10(np.mean(np.sum((x1_test_r-x1_hat_r)**2,axis=1)/np.sum(x1_test_r**2,axis=1)))
  # err = 10*np.log10(np.mean((np.sum((x_test-xhat)**2,axis=1)/np.sum(x_test**2,axis=1))))
  
  
  print('Name:', name)
  print('Error:', err, 'dB')
  if show_plots == True:
    for i in range(10):
      plt.figure()
      plt.plot(x1_hat_r[i], 'bo', label='xhat')
      plt.plot(x1_test_r[i], 'ro', label='x_test')
      plt.legend()
      plt.show()

  return err

def train_various_param_inits(props, L, params_initialization, data_train,data_test, old_net=None):
  L1,L2,L3,L4 = L
  lam,lam2,alph,rho = params_initialization

  names = np.empty((L1,L2,L3,L4),dtype=object)
  nets = np.empty((L1,L2,L3,L4),dtype=object)
  
  for l1 in range(L1):
    for l2 in range(L2):
      for l3 in range(L3):
        for l4 in range(L4):
          params_init = {'lambda':lam[l3] , 'lambda2':lam2[l4], 'alpha':alph[l2] , 'rho':rho[l1], 'beta':1.}
          if old_net is None:
            a = tn.gen_net(props, 
                        'untied',
                        params_init=params_init
                )
          else:
            a = old_net
          
          a = tn.train_net(a, data_train, props)
          names[l1,l2,l3,l4] = tn.save_net(a,props)
          nets[l1,l2,l3,l4] = a
             
  return nets, names

def load_net(folder, scen, num_layers):
  # args = ln.get_args()
  # scen = args[0].scen
  # folder = './nets/' + args[0].folder
  folder = './nets/' + folder

  A = np.load(folder + '/A.npy')
  n1,n2 = A.shape
  p = problem.Problem((n1,n2-n1),scen,partition=True, N_part=n2-n1)
  
  a = admm.ADMMNet(p, num_layers, 'untied')
  # a = lista.ISTANet('blank', problem=p)
  a.load_weights(folder + '/weights')

  return a, p

def train_old():
  name = 'mimo_d_20x60_SNR=35dB_10stages_s1=2_s2=6_admm_untied_05_10_2020_11_47_49'
  scenario = 'mimo_d'
  num_stages = 10

  m = 20
  s1 = 2
  s2 = 6
  SNR = 35
  lam = 1e-2; lam2 = 1e-3; alph=1.7e-0; rho = 4e-2
  params_initialization = {'lambda':lam , 'lambda2':lam2, 'alpha':alph, 'rho':rho, 'beta':1.}
  append_layer = True

  a, p = load_net(name, scenario, num_stages)

  p, data_train, data_test = problem_setup(
                                dims=(m,2*m),
                                SNR=SNR, 
                                scen=scenario, 
                                partition=True, 
                                Ntrain=int(1e5), 
                                Ntest=int(1e5),
                                sparsity1=s1,
                                sparsity2=s2)
            
  props = dict(net_type='admm', 
            num_stages=num_stages, 
            problem=p,
            learning_rate=1e-3,
            optimizer='adam',
            loss='MSE',
            schedule=False,
            Ntrain=data_train[0].shape[0], #num training samples
            batch_size=1000, # batch size
            epochs=20,
            SNR=SNR,
            s1=s1,
            s2=s2
        )

  if append_layer == True:
    a.Layers.append(admm.Stage(params_initialization, p))
  
  a = tn.train_net(a, data_train, props)

  err = eval_net(a, data_test, name, quiet=False)
  new_name = tn.save_net(a,props)

  return a, new_name

def problem_setup(dims, SNR, scen, partition, Ntrain, Ntest, sparsity1=None, sparsity2=None, old_p=None):
  m,n = dims
  if old_p is None:
    p = problem.Problem((m,n), scen, partition=partition, N_part=n)
  else:
    p = old_p
  m,n = p.A.shape
  n = n - m
  # x,y = gen_samples(p.A, 1e5, density=0.2, normalize=False, ones=False, comp=True)

  train_number = Ntrain
  test_number = Ntest
  sparsity = 2
  # sparsity1 = 2
  # sparsity2 = 1

  if partition == False:
    y,x = generate_data(p.A, sparsity, train_number, SNR=SNR, mismatch=False)
  elif partition == True:
    y,x = generate_data_with_partition(p.A, n, sparsity1, sparsity2, train_number, SNR=SNR, noise=True, mismatch=False)
  y = np.concatenate((y.real,y.imag),axis=1)
  x = np.concatenate((x.real,x.imag),axis=1)

  if partition == False:
    y_test,x_test = generate_data(p.A, sparsity, train_number, SNR=SNR, mismatch=False)
  elif partition == True:
    y_test,x_test = generate_data_with_partition(p.A, n, sparsity1, sparsity2, test_number, SNR=SNR, noise=True, mismatch=False)
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

def train_new():
  # lambda lambda2 alpha rho
  # L3 = 1; L2 = 1; L1 = 1
  L4 = 1; L3 = 1; L2 = 1; L1 = 1
  # lam = [2e-2]; alph=[1.8]; rho = [1.] # 1 + 1j only
  # lam = [1e-2]; lam2 = [4e-3]; alph=[1.8e-0]; rho = [1e-0]
  lam = [1e-1]; lam2 = [1.78e-2]; alph=[1.7e-0]; rho = [1e-2]
  lam = [1e-2]; lam2 = [1e-3]; alph=[1.7e-0]; rho = [4e-2]
  # lam = np.logspace(-2.5,-1.5,L3, dtype=np.float32)
  # lam2 = np.logspace(-4,-3,L4, dtype=np.float32)
  # lam2 = np.linspace(5e-3,5e-2,L4, dtype=np.float32)

  # alph = 1*np.logspace(-2,-0.,L2, dtype=np.float32)
  # alph = np.linspace(1.5,1.8,L2,dtype=np.float32)

  # rho = np.logspace(-2,0,L1, dtype=np.float32)
  # rho = np.linspace(1,1.9,L1,dtype=np.float32)

  L = (L1,L2,L3,L4)
  params_initialization = (lam,lam2,alph,rho)

  nets_list = []
  names_list = []
  errs_list = []

  SNR_list = range(0,21,5)
  # SNR_list = [15]
  # files = ['siso_d_8x23_SNR=10dB_10stages_admm_untied_04_27_2020_13_30_22', 'siso_d_8x23_SNR=20dB_10stages_admm_untied_04_27_2020_13_33_48', 'siso_d_8x23_SNR=30dB_10stages_admm_untied_04_27_2020_13_37_32', 'siso_d_8x23_SNR=40dB_10stages_admm_untied_04_27_2020_13_40_56']
  # num_stages_list = range(5,16,2)
  num_stages_list = [1,2,3]

  # sparsities = list(zip(range(2,9),7*[1]))
  # sparsities = [(2,8),(2,10),(2,12),(2,14)]
  sparsities = [(4,6),(6,6),(8,6),(10,6)]
  s1 = 2
  s2 = 6
  # SNR = 20
  m = 20
  scenario = 'siso'
  for SNR in [30]: #[15,20,25,30,35]: #SNR_list:
    # print('\n-----------------------------------SNR =',SNR,'--------------------------------')
    # for m in [20]: # range(5,16,2):
    #       print('\n-----------------------------------m =',m,'--------------------------------')
    # for s1,s2 in sparsities:
    # for scenario in ['siso', 'siso_d', 'mimo_d']:
          print('s1=',s1,'s2=',s2,'\n')
          p, data_train, data_test = problem_setup(
                                                  dims=(m,2*m),
                                                  SNR=SNR, 
                                                  scen=scenario, 
                                                  partition=True, 
                                                  Ntrain=int(1e6), 
                                                  Ntest=int(1e5),
                                                  sparsity1=s1,
                                                  sparsity2=s2)
          for num_stages in num_stages_list:
            
              props = dict(net_type='admm', 
                        num_stages=num_stages, 
                        problem=p,
                        learning_rate=1e-3,
                        optimizer='adam',
                        loss='MSE',
                        schedule=False,
                        Ntrain=data_train[0].shape[0], #num training samples
                        batch_size=1000, # batch size
                        epochs=20,
                        SNR=SNR,
                        s1=s1,
                        s2=s2
                    )

              # old_net = load_net('siso_10x30_SNR=15dB_10stages_s1=2_s2=3_admm_untied_05_08_2020_01_28_24', 'siso', loss='NMSE')

              nets, names = train_various_param_inits(props, L, params_initialization, data_train, data_test) #, old_net)
              
              # nets = train_old_net()

              nets_list.append(nets)
              names_list.append(names)
              errs_list.append(eval_nets(nets, data_test, L, params_initialization, names, quiet=False))


  for names,errs in zip(names_list,errs_list):
    print(names)
    print(errs)

def eval_old_net():
  p, data_train, data_test = problem_setup(
                                                    dims=(20,40),
                                                    SNR=30, 
                                                    scen='mimo_d', 
                                                    partition=True, 
                                                    Ntrain=0, 
                                                    Ntest=int(1e5),
                                                    sparsity1=2,
                                                    sparsity2=6)
  # name = 'siso_20x60_SNR=15dB_10stages_s1=2_s2=6_admm_untied_05_09_2020_23_16_26'
  # name = 'siso_20x60_SNR=15dB_10stages_s1=3_s2=6_admm_untied_05_09_2020_13_21_45'
  name = 'siso_20x60_SNR=15dB_10stages_s1=2_s2=6_admm_untied_05_09_2020_08_40_28'
  name = 'siso_20x60_SNR=15dB_10stages_s1=2_s2=6_admm_untied_05_10_2020_00_31_22'
  name = 'mimo_d_20x60_SNR=35dB_10stages_s1=2_s2=6_admm_untied_05_10_2020_11_47_49'
  name = 'mimo_d_20x60_SNR=30dB_3stages_s1=2_s2=6_admm_untied_05_11_2020_02_10_32'
  net,p = load_net(name, 'mimo_d', )
  eval_net(net, data_test, name, False)

def train_new_iteratively(SNR, scenario):
  # name = 'mimo_d_20x60_SNR=35dB_10stages_s1=2_s2=6_admm_untied_05_10_2020_11_47_49'
  # scenario = 'mimo_d'
  num_stages = 1

  m = 20
  s1 = 2
  s2 = 6
  # SNR = 30
  lam = 1e-2; lam2 = 1e-3; alph=1.7e-0; rho = 4e-2
  params_initialization = {'lambda':lam , 'lambda2':lam2, 'alpha':alph, 'rho':rho, 'beta':1.}

  p, data_train, data_test = problem_setup(
                                dims=(m,2*m),
                                SNR=SNR, 
                                scen=scenario, 
                                partition=True, 
                                Ntrain=int(1e6), 
                                Ntest=int(1e5),
                                sparsity1=s1,
                                sparsity2=s2)
            
  props = dict(net_type='admm', 
            num_stages=num_stages, 
            problem=p,
            learning_rate=1e-3,
            optimizer='adam',
            loss='MSE',
            schedule=False,
            Ntrain=data_train[0].shape[0], #num training samples
            batch_size=1000, # batch size
            epochs=20,
            SNR=SNR,
            s1=s1,
            s2=s2
        )

  a = tn.gen_net(props, 
                 'untied',
                  params_init=params_initialization
                )

  for i in range(15):
    if i > 0:
      a.Layers.append(admm.Stage(params_initialization, p))
    a = tn.train_net(a, data_train, props)
    disp_msg = 'stages = ' + str(num_stages+i)
    err = eval_net(a, data_test, disp_msg, show_plots=False)

  props['num_stages'] = num_stages + i
  new_name = tn.save_net(a,props)
  print(new_name)

  # a, p = load_net(name, scenario, num_stages)

  return a, new_name
# train_new()
# train_old()
# eval_old_net()
# for scenario in ['mimo_d']:
#   for SNR in [30]:
#     train_new_iteratively(SNR, scenario)