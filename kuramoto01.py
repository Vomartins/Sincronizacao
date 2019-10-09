import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import newton_krylov
#Criar uma pasta chamada frames01 onde o programa estiver salvo para as figuras

#Modelo de Kuramoto
def kuramoto(t, y, K, N, A, W, w):
    x = y
    
    dydt = np.zeros(3*N)    
    
    for i in range(N):
        rho = np.zeros(3)
        a = np.array([x[i],x[i+N],x[i+2*N]])
        for j in range(N):
            b = np.array([x[j],x[j+N],x[j+2*N]])
            rho = rho + A[i,j]*b
        rho = (1/N)*rho
        
        dydt[i] = K*(rho[0] - np.inner(rho,a)*x[i]) + np.cross(W[i]*w[:,i],a)[0]
        dydt[i+N] = K*(rho[1] - np.inner(rho,a)*x[i+N]) + np.cross(W[i]*w[:,i],a)[1]
        dydt[i+2*N] = K*(rho[2] - np.inner(rho,a)*x[i+2*N]) + np.cross(W[i]*w[:,i],a)[2]
        
    return dydt

def kuramotoPF(y, K, N, A, W, w):
    x = y
    
    dydt = np.zeros(3*N)    
    
    for i in range(N):
        rho = np.zeros(3)
        a = np.array([x[i],x[i+N],x[i+2*N]])
        for j in range(N):
            b = np.array([x[j],x[j+N],x[j+2*N]])
            rho = rho + A[i,j]*b
        rho = (1/N)*rho
        
        dydt[i] = K*(rho[0] - np.inner(rho,a)*x[i]) + np.cross(W[i]*w[:,i],a)[0]
        dydt[i+N] = K*(rho[1] - np.inner(rho,a)*x[i+N]) + np.cross(W[i]*w[:,i],a)[1]
        dydt[i+2*N] = K*(rho[2] - np.inner(rho,a)*x[i+2*N]) + np.cross(W[i]*w[:,i],a)[2]
        
    return dydt
#Parâmetros.
N = 10
s = [0, 100]
K = 0.8
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
            
#condição inicial
theta = np.random.uniform(0, 2*np.pi, N) 
phi = np.random.uniform(0, np.pi, N)

x0 = np.cos(theta)*np.sin(phi)
y0 = np.sin(theta)*np.sin(phi)
z0 = np.cos(phi)

init_state = np.append(x0, [y0 , z0])
#Solução do modelo
sol = solve_ivp(lambda t, y: kuramoto(t, y, K, N, A, W, w), s, init_state)
#pontos fixos
chute_inicial = init_state
pontos_fixos = newton_krylov(lambda y: kuramotoPF(y, K, N, A, W, w),chute_inicial)

x_f = pontos_fixos[0:N]
y_f = pontos_fixos[N:2*N]
z_f = pontos_fixos[2*N:3*N]

d = np.zeros(N)
for i in range(N):
    d[i] = x_f[i]**2 + y_f[i]**2 + z_f[i]**2
print(d)

for i in range(int(sol.y.shape[1])):
    #Plot dos frames
    
    x = sol.y[0:N,i]
    y = sol.y[N:2*N,i]
    z = sol.y[2*N:3*N,i]

    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 30)

    u, v = np.meshgrid(theta, phi)

    X = np.cos(u)*np.sin(v)
    Y = np.sin(u)*np.sin(v)
    Z = np.cos(v)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection ='3d')
    ax.plot_wireframe(X, Y, Z, color='0.75', alpha='0.4')
    ax.scatter(x_f,y_f,z_f, c='b', s=50)
    
    for j in range(N):
        ax1 = fig.gca(projection='3d')
        ax1.scatter(x[j], y[j], z[j], c='k',s=50)
    
    plt.suptitle('{} individuals - t={}.'.format(N,np.round(sol.t[i],2)), size=40)
    
    if i < 10:
        plt.savefig('frames01/00{}.png'.format(int(i)))
    elif 9 < i < 100:
        plt.savefig('frames01/0{}.png'.format(i))
    else:
        plt.savefig('frames01/{}.png'.format(int(i)))
    plt.close()