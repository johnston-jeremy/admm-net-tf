import numpy as np
import matplotlib.pyplot as plt

x = np.exp(1j*2*np.pi*0.1*np.arange(10)) + np.exp(1j*2*np.pi*0.3*np.arange(10))

N = 10
tau_e = 0.1
X = np.fft.fft(x,n=N)
Xe = X*np.exp(1j*2*np.pi*tau_e*np.arange(N))
xe = np.fft.ifft(Xe)

phi_x = np.arctan(x.imag/x.real)
phi_xe = np.arctan(xe.imag/xe.real)
plt.plot(phi_x-phi_xe)
# plt.plot(phi_xe)
plt.show()