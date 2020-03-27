# import numpy as np

# def khatri_rao(a, b):
#     c = a[...,:,np.newaxis,:] * b[...,np.newaxis,:,:]
#     # collapse the first two axes
#     return c.reshape((-1,) + c.shape[2:])

# Np = 8 # num pulses
# Ntx = 2
# Nrx = 2
# M = Np*Ntx*Nrx

# Ngl = 16;
# Ngk = 8;
# A1 = np.zeros((M,Ngl*Ngk));

# for l in range(Ngl):
#   for k in range(Ngk):
#       # v = exp(1j*2*pi*(l/Ngl)*(0:N-1)');
#       # H = exp(1j*2*pi*(k/Ngk)*(0:Nrx-1)')*exp(1j*2*pi*(k/Ngk)*(0:Ntx-1));

#       v = np.cos(np.pi*((l/Ngl)*np.arange(Np).T))
#       H = np.real(np.outer(np.exp(1j*np.pi*(k/Ngk)*np.arange(Nrx)),np.exp(1j*np.pi*(k/Ngk)*np.arange(Nrx))))

#       h = H.ravel()
#       A1[:, l*Ngk + k] = np.kron(v,h)


# Ntx = 2
# Nrx = 2
# # M = Np*Ntx*Nrx
# # # N = np.round(np.sqrt(2*M))
# # N = 4*M
# Ngl = 16
# Ngk = 8

# # m = np.arange(M)[:,None]
# ngl = np.arange(Ngl)
# ngk = np.arange(Ngk)

# v = np.cos(np.pi*np.outer(np.arange(Np),ngl)/Ngl)
# hrx = np.cos(np.pi*np.outer(np.arange(Nrx),ngk)/Ngk)
# htx = np.cos(np.pi*np.outer(np.arange(Ntx),ngk)/Ngk)
# h = khatri_rao(hrx,htx)
# A2 = np.kron(h,v)

# A = A1
# Atop = np.concatenate((np.real(A),-np.imag(A)),axis=1)
# Abot = np.concatenate((np.imag(A),np.real(A)),axis=1)
# A1 = np.concatenate((Atop,Abot),axis=0)

# A = A2
# Atop = np.concatenate((np.real(A),-np.imag(A)),axis=1)
# Abot = np.concatenate((np.imag(A),np.real(A)),axis=1)
# A2 = np.concatenate((Atop,Abot),axis=0)

# print(A1.shape)
# print(A2.shape)

import scipy as sp
import scipy.io
import numpy as np
density = .5
A = np.random.rand(2,2)
N = 3
n1,n2 = A.shape
x = sp.sparse.random(n2,N,density,dtype=np.float32).A
y = np.matmul(A,x)

x = x.T
y = y.T
# print(np.sum(np.where(x!=0,1,0)))

y1 = np.zeros((N,n1))
x1 = np.zeros((N,n2))
for i in range(N):
  x1[i] = sp.sparse.random(n2,1,density,dtype=np.float32).A.T[0]
  # if normalize == True:
  #   x[i] = x[i]/np.sqrt(np.sum(x[i]**2))
  # if ones == True:
  #   x[i] = np.divide(x[i],x[i],out=np.zeros_like(x[i]),where=x[i]!=0)
  y1[i] = np.matmul(A,x[i]).T

print(np.sum(y1==y))
print(type(y))
print(type(y1))
breakpoint()
# print(np.sum(y==y1))
# print(y)
# print(y1)