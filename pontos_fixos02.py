import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import newton_krylov

def  kuramoto(y, K, N, A, W, w):
    x = y
        
    dydt = np.zeros(3*N)
    for i in range(N):
        a = np.array([x[i], x[i+N], x[i+2*N]])
        for j in range(N):
            b = np.array([x[j], x[j+N], x[j+2*N]])
            s_0 = x[j] - np.inner(b,a)*x[i]
            s_1 = x[j+N] - np.inner(b,a)*x[i+N]
            s_2 = x[j+2*N] - np.inner(b,a)*x[i+2*N]
        dydt[i] = (K/N)*s_0 + np.cross(W[i]*w[:,i],a)[0]
        dydt[i+N] = (K/N)*s_1 + np.cross(W[i]*w[:,i],a)[1]
        dydt[i+2*N] = (K/N)*s_2 + np.cross(W[i]*w[:,i],a)[2]
    return dydt

N = 30
s = [0, 300]
K = 0.1
mu = 0
delta = 0.5
A = np.full((N,N), 1)
W = np.random.normal(mu, delta, N)
w = np.zeros((3,N))
for k in range(N):
    H = np.full((3,3), W[k])
    for l in range(3):
        H[l, l] = 0 
    for l in range(3):
        for m in range(3):
            if l>m:
                H[l,m] = -H[l,m]
    L, V = np.linalg.eig(H)
    for l in range(3):
        if np.real(L[l])<(10**(-6)) and np.imag(L[l])<(10**(-6)):
            w[:,k] = V[:,l]
            
theta = np.random.uniform(0, 2*np.pi, N) 
phi = np.random.uniform(0, np.pi, N)

x0 = np.cos(theta)*np.sin(phi)
y0 = np.sin(theta)*np.sin(phi)
z0 = np.cos(phi)

chute_inicial = np.append(x0, [y0 , z0])
pontos_fixos = newton_krylov(lambda y: kuramoto(y, K, N, A, W, w),chute_inicial)

x = pontos_fixos[0:N]
y = pontos_fixos[N:2*N]
z = pontos_fixos[2*N:3*N]

theta = np.linspace(0, 2*np.pi, 30)
phi = np.linspace(0, np.pi, 30)

u, v = np.meshgrid(theta, phi)

X = np.cos(u)*np.sin(v)
Y = np.sin(u)*np.sin(v)
Z = np.cos(v)

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection ='3d')
ax.plot_wireframe(X, Y, Z, color='0.75', alpha='0.4')
ax.scatter(x,y,z, c='b', s=50)
plt.show()