import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import newton_krylov
#Criar uma pasta chamada frames02 onde o programa estiver salvo para as figuras

#Modelo de Kuramoto
def kuramoto(t, y, K, N, A, W, w):
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

#Parâmetros.
N = 30
s = [0, 300]
K = 2
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

#Construção do vetor rho e dos vetores ponto fixo
rho = np.zeros(3)
for k in range(N):
    rho = rho + np.array([sol.y[k,-1],sol.y[k+N,-1],sol.y[k+2*N,-1]])
rho = (1/N)*rho
nrho = np.linalg.norm(rho)
rho = rho/nrho

mi = np.zeros(N)
for k in range(N):
    mi[k] = W[k]/(K*nrho)

tal = np.zeros((N,2))
for i in range(N):    
    tal[i, 0] = + (((1 - mi[i]**2)+((mi[i]**2 - 1)**2+4*(mi[i]**2)*(np.dot(rho, w[:,i]))**2)**(1/2))/2)**(1/2)
    tal[i, 1] = - (((1 - mi[i]**2)+((mi[i]**2 - 1)**2+4*(mi[i]**2)*(np.dot(rho, w[:,i]))**2)**(1/2))/2)**(1/2)

ep = np.zeros((N, 2))
for i in range(N):
    ep[i, 0] = np.inner(rho, w[:,i])/tal[i, 0]
    ep[i, 1] = np.inner(rho, w[:,i])/tal[i, 1]

sigmaf = np.zeros((N,6))
for i in range(N):
    sigmaf[i,0:3] = (1/(1+(ep[i,0]**2)*(mi[i]**2)))*((mi[i]*np.cross(w[:,i],rho))+(ep[i,0]*(mi[i]**2)*w[:,i])+(tal[i,0]*rho))
    sigmaf[i,3:6] =(1/(1+(ep[i,1]**2)*(mi[i]**2)))*((mi[i]*np.cross(w[:,i],rho))+(ep[i,1]*(mi[i]**2)*w[:,i])+(tal[i,1]*rho))

for i in range(int(sol.y.shape[1])): #t+1
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
    ax.scatter(sigmaf[:,0],sigmaf[:,1],sigmaf[:,2], c='b', s=50)
    ax.scatter(sigmaf[:,3],sigmaf[:,4],sigmaf[:,5], c='r', s=50)
    #ax0 = fig.gca(projection='3d')
    #ax0.scatter(rho[0], rho[1], rho[2], c='r',s=50)
    
    for j in range(N):
        ax1 = fig.gca(projection='3d')
        ax1.scatter(x[j], y[j], z[j], c='k',s=50)
    
    plt.suptitle('{} individuals - t={}.'.format(N,np.round(sol.t[i],2)), size=40)
    
    if i < 10:
        plt.savefig('frames02/00{}.png'.format(int(i)))
    elif 9 < i < 100:
        plt.savefig('frames02/0{}.png'.format(i))
    else:
        plt.savefig('frames02/{}.png'.format(int(i)))
    plt.close()