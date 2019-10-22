import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import newton_krylov

def kuramoto(y, K, N, A, W, w):
    x = y
    
    d = np.zeros(N)
    for i in range(N):
        for j in range(N):
            d[i] = d[i] + A[i][j] 
    
    dydt = np.zeros(3*N)    
    
    for i in range(N):
        rho = np.zeros(3)
        a = np.array([x[i],x[i+N],x[i+2*N]])
        for j in range(N):
            b = np.array([x[j],x[j+N],x[j+2*N]])
            rho = rho + A[i,j]*b
        rho = (1/N)*rho
        #rho = (1/d[j])*rho
        
        dydt[i] = K*(rho[0] - (np.inner(rho,a))*x[i]) + np.cross(W[i]*w[:,i],a)[0]

        dydt[i+N] = K*(rho[1] - (np.inner(rho,a))*x[i+N]) + np.cross(W[i]*w[:,i],a)[1]

        dydt[i+2*N] = K*(rho[2] - (np.inner(rho,a))*x[i+2*N]) + np.cross(W[i]*w[:,i],a)[2]
        
    return dydt

N = 4
s = [0, 50]
K = 1
mu = 0
delta = 0.1
A = np.full((N,N), 1)
W = np.random.normal(mu, delta, N)
w = np.zeros((3,N))
for i in range(N):
    H = np.full((3,3), W[i])
    for j in range(3):
        H[j, j] = 0 
    for j in range(3):
        for k in range(3):
            if j>k:
                H[j,k] = -H[j,k]
    L, V = np.linalg.eig(H)
    for j in range(3):
        if np.real(L[j])<(10**(-6)) and np.imag(L[j])<(10**(-6)):
            w[:,i] = V[:,j]
            
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