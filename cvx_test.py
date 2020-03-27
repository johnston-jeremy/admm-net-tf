import cvxpy as cp
import numpy
import matplotlib.pyplot as plt
import problem
import tf_classes as mytf
import tensorflow as tf
import numpy as np
import scipy as sp
import time
import numpy.linalg as la
import argparse
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
from train_net import gen_samples
import load_net

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Assign a value to gamma and find the optimal x.
def get_x(y,p):
  # Construct the problem.
  x = cp.Variable(p.size(1))
  gamma = cp.Parameter(nonneg=True)
  error = cp.sum_squares(p.A @ x - y)
  obj = cp.Minimize(error + gamma*cp.norm(x, 1))
  prob = cp.Problem(obj)
  gamma.value = p.lambda_init
  result = prob.solve(solver=cp.ECOS)
  return x.value


def cvx_test_mp(p, y):
  pool = Pool(cpu_count())
  iterable = zip(y,[p]*y.shape[0])
  return pool.starmap(get_x, iterable,chunksize=256)

def cvx_test(p, y):
  x = np.zeros((y.shape[0],p.size(1)))
  for i in range(y.shape[0]):
    x[i] = get_x(y[i], p)
  return x


# # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
# sq_penalty = []
# l1_penalty = []
# x_values = []
# gamma_vals = numpy.logspace(-4, 6)
# for val in gamma_vals:
#     gamma.value = val
#     prob.solve()
#     # Use expr.value to get the numerical value of
#     # an expression in the problem.
#     sq_penalty.append(error.value)
#     l1_penalty.append(cp.norm(x, 1).value)
#     x_values.append(x.value)

if __name__ == "__main__":
  args = load_net.get_args()
  scen = args[0].scen
  n1 = 250
  n2 = 500
  p = problem.Problem((n1,n2),scen)
  N = 10
  x,y = gen_samples(p.A,N)
  # gamma must be nonnegative due to DCP rules.
  gamma = p.lambda_init
  from time import time
  t = time()
  xhat = cvx_test_mp(p,y)
  print('Avg Runtime:', (time()-t)/N)
  print('Avg Error:',np.mean((np.sum((x-xhat)**2,axis=1)/np.sum(x**2,axis=1))))
  # # Construct the problem.
  # x = cp.Variable(n1)
  # error = cp.sum_squares(A @ x - b)
  # obj = cp.Minimize(error + gamma*cp.norm(x, 1))
  # prob = cp.Problem(obj)
  # prob.solve()
  # x_values.append(x.value)
