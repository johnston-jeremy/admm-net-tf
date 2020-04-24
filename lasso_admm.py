import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import random as rd
# import cvxpy as cp

def aula(A,rho0):
  M = A.shape[0]
  N = A.shape[1]
  L = la.cholesky(np.eye(M) + (1/rho0)*np.matmul(A,A.T.conj()))
  U = L.T.conj();
  #AULA = (A'*(U \ ( L \ A )));
  return np.matmul(A.T.conj(),np.matmul(la.inv(U),np.matmul(la.inv(L),A)))

def factor(A,rho):
  M = A.shape[0]
  L = la.cholesky(np.eye(M) + (1/rho)*np.matmul(A,A.T.conj()))
  U = L.T.conj()
  return L,U
  
def lasso_admm(A, y, lam, rho, alpha, eta, maxiter):

  m,n = A.shape
  z = np.zeros(n,dtype=np.complex128)
  u = np.zeros(n,dtype=np.complex128)

  Aty = np.matmul(A.T.conj(),y)
  L,U = factor(A,rho)

  AULA = aula(A, rho)
  
  M1 = np.matmul(np.eye(n)/rho - (1/rho**2)*AULA, A.T.conj())
  M2 = (1/rho)*np.eye(n) - (1/rho**2)*AULA

  zhistory = np.zeros((maxiter,n),dtype=np.complex128)
  for k in range(maxiter):
    # x-update
    x = np.matmul(M1,y) + np.matmul(M2,rho*(z - u))

    # z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold
    z = softthreshold(x_hat + u, lam/rho)

    # u-update
    u = u + eta*(x_hat - z)

    zhistory[k] = z
  # plt.show()
  return z, zhistory

def generate_data(A, sparsity, bs, SNR=20):
    import random as rd
    m, n = A.shape
    x_real = np.zeros([n, bs])
    x_imag = np.zeros([n, bs])
    x = np.zeros([n, bs]) + 1j * np.zeros([n, bs])
    for i in range(bs):
        idx = rd.sample(list(range(n)), sparsity)
        
        # x_real[idx, i] = np.ones(sparsity)
        # x_imag[idx, i] = np.ones(sparsity)
        # temp = x_real[:, i] + 1j * x_imag[:, i]
        # x[:, i] = temp                                     # without the normalization of x
        # x[idx,i] = np.exp(1j*2*np.pi*np.random.rand(sparsity))
        x[idx,i] = 2 * ( np.random.rand(sparsity) + 1j*np.random.rand(sparsity) ) - 1 - 1j

    X = x.copy()

    # ----- Generate the compressed noiseless signals --------
    Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    for i in range(bs):
        tmp = np.matmul(A, X[:, i])
        # tmp /= np.linalg.norm(tmp)
        Y[:, i] = tmp

    # ----- Generate the compressed signals with noise -------
    # noise_std = np.power(10, -(SNR / 20))
    # Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    # for i in range(bs):
    #     tmp = np.matmul(A, X[:, i])
    #     # tmp /= np.linalg.norm(tmp)
    #     Y[:, i] = tmp + noise_std * (rng.randn(m) + 1j * rng.randn(m))
    return Y, X

def gen_samples(A,N,sparsity,normalize=False,comp=True):
  N = int(N)
  n1,n2 = A.shape

  if comp==True:
    x = np.zeros((n2,N),dtype=np.complex128)
    for i in range(N):
      r = np.random.rand(sparsity)
      ind = np.random.permutation(np.arange(n2))[:sparsity]
      # x[ind,i] = np.exp(1j*2*np.pi*r)
      x[ind,i] = 2 * (np.random.rand(sparsity) + 1j*np.random.rand(sparsity)) - 1 - 1j

  y = np.matmul(x,A)

  return x,y

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

def lasso_cvx(A,y,gam):
  # Construct the problem.
  x = cp.Variable(p.size(1), complex=True)
  z = cp.Variable(p.size(0), complex=True)
  gamma = cp.Parameter(nonneg=True)
  error = cp.sum_squares(A @ x - y)
  constraints = []

  obj = cp.Minimize(error + gamma*cp.norm(x, 1))

  # obj = cp.Minimize(0.5*cp.norm2(cp.abs(z))**2 + gamma*cp.norm1(x))
  # constraints = [z == (A @ x - y)]

  prob = cp.Problem(obj, constraints)
  gamma.value = gam
  result = prob.solve(solver=cp.ECOS)
  return x.value

def softthreshold(x,kappa):
  xnormalized = np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x)!=0)
  return xnormalized*np.maximum(np.abs(x) - kappa,0)

if __name__ == '__main__':
# def main():
  import problem

  # Number of sample vectors
  Nsamp = 100

  # p = problem.Problem('',(10,20))
  # M,Ng = A.shape
  # # sparsity
  # sparsity = 2
  # Y,X = generate_data(A, sparsity, Nsamp)
  # # X,Y = gen_samples(A,Nsamp,sparsity)

  N_part = 15
  p = problem.Problem((10,20), 'siso partition', N_part=N_part)

  # x,y = gen_samples(p.A, 1e5, density=0.2, normalize=False, ones=False, comp=True)

  train_number = int(1e5)
  valid_number = 1000
  sparsity = 2
  sparsity1 = 2
  sparsity2 = 1

  # y,x = generate_data(p.A, sparsity, train_number, SNR=100, noise=True)
  y,x = generate_data_with_partition(p.A, N_part, sparsity1, sparsity2, train_number, SNR=100, noise=False)

  Y=y.T
  X=x.T

  A = p.A
  # ADMM setup
  # number of ADMM iterations
  maxiter = 100;
  # algorithm parameters
  L = [1,1,3,1] # [rho, alpha, lambda]
  # L = [3,3,3,3] # [rho, alpha, lambda]
  # rho = 1.*np.logspace(-2,-0,L[0])
  alph = 1.8*np.logspace(-2,0,L[1]);
  lam = np.logspace(-3,-1,L[2]);
  eta = 1.*np.logspace(-2,-0,L[0])

  # lam = [1e-3]
  rho = [1e-0]
  # alph = [1.8]

  # To store results
  errs = np.zeros((L[0],L[1],L[2],L[3]))
  its = np.zeros((L[0],L[1],L[2],L[3]))
  admmiters = np.zeros(Nsamp)
  err_single = np.zeros(Nsamp)

  print('A shape', A.shape)
  print('sparsity =',sparsity)
  print('maxiter =', maxiter)
  print('lambda\t\t','rho\t\t','alpha\t\t','eta\t\t','avg_err')
  print('----------------------------------------------------------------------------')
  for r in range(L[0]):
      for a in range(L[1]):
          for l in range(L[2]):
            for e in range(L[3]):
              for i in range(Nsamp):
                  
                y = Y[:,i]
                x = X[:,i]

                x_hat, x_history = lasso_admm(A, y, lam[l], rho[r], alph[a], eta[e], maxiter)

                # x_hat = lasso_cvx(A, y, lam[l])

                err_single[i] = (la.norm(x-x_hat)**2)/(la.norm(x)**2);
                  
              errs[r,a,l,e] = 10*np.log10(np.mean(err_single));

              print('{:.2e}\t'.format(lam[l]), '{:.2e}\t'.format(rho[r]), '{:.2e}\t'.format(alph[a]), '{:.2e}\t'.format(eta[e]),'{:.2e}'.format(errs[r,a,l,e]))
              
  
  err_best = np.min(errs);
  a,b,c,d = np.unravel_index(np.argmin(errs),errs.shape)
  print('err(dB)\tlambda\t\trho\t\talpha\t\teta')
  print('{:.2f}\t'.format(err_best), '{:.2e}\t'.format(lam[c]), '{:.2e}\t'.format(rho[a]), '{:.2e}\t'.format(alph[b]), '{:.2e}\t'.format(eta[d]))
  


