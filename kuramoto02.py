import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
#Criar uma pasta chamada frames02 onde o programa estiver salvo para as figuras

#Modelo de Kuramoto
def kuramoto(t, y, K, N, A, W):
    sigma = y
    
    d = np.zeros(N)
    for i in range(N):
        for j in range(N):
            d[i] = d[i] + A[i][j] 
    
    dydt = np.zeros(3*N)    
    
    for i in range(N):
        w = W[i]
        rho = np.zeros(3)
        for j in range(N):
            p = np.inner(np.array([sigma[j],sigma[j+N],sigma[j+2*N]]),np.array([sigma[i],sigma[i+N],sigma[i+2*N]]))
            dydt[i] = dydt[i] + A[i][j]*(sigma[j] - p*sigma[i])
            dydt[i+N] = dydt[i+N] + A[i][j]*(sigma[j+N] - p*sigma[i+N])
            dydt[i+2*N] = dydt[i+2*N] + A[i][j]*(sigma[j+2*N] - p*sigma[i+2*N])
        
        dydt[i] = (K/N)*dydt[i] + w*(sigma[i+2*N] - sigma[i+N])
        dydt[i+N] = (K/N)*dydt[i+N] + w*(sigma[i] - sigma[i+2*N])
        dydt[i+2*N] = (K/N)*dydt[i+2*N] + w*(sigma[i+N] - sigma[i])
        
    return dydt

#Parâmetros.
N = 10
t = 30
T = 3000
s = np.arange(0, T, t)
K = 2
mu = 0
delta = 0.1
A = np.full((N,N), 1)
W = np.random.normal(mu, delta, N)

#condição inicial
theta = np.random.uniform(0, 2*np.pi, N) 
phi = np.random.uniform(0, np.pi, N)

x0 = np.cos(theta)*np.sin(phi)
y0 = np.sin(theta)*np.sin(phi)
z0 = np.cos(phi)

init_state = np.append(x0, [y0 , z0])
#Solução do modelo
sol = solve_ivp(lambda t, y: kuramoto(t, y, K, N, A, W), s, init_state, method='BDF')

#Construção do vetor rho e dos vetores ponto fixo
rho = np.zeros(3)
for k in range(N):
    rho = rho + np.array([sol.y[k,0],sol.y[k+N,0],sol.y[k+2*N,0]])

nrho = np.linalg.norm(rho)
rho = rho/nrho

mi = np.zeros(N)
for k in range(N):
    mi[k] = W[k]/(K*nrho)

w = np.zeros((N, 3))
for i in range(N):
    w[i,:] = np.array([W[i], W[i], W[i]])/np.linalg.norm(np.array([W[i], W[i], W[i]]))

tal = np.zeros((N,2))
for i in range(N):    
    tal[i, 0] = + ((((1-(mi[i]**2))+(((((mi[i]**2)-1)**2)+4*(mi[i]**2)*(np.inner(rho, w[i,:])**2))**(1/2)))/2)**(1/2))
    tal[i, 1] = - ((((1-(mi[i]**2))+(((((mi[i]**2)-1)**2)+4*(mi[i]**2)*(np.inner(rho, w[i,:])**2))**(1/2)))/2)**(1/2))

ep = np.zeros((N, 2))
for i in range(N):
    ep[i, 0] = np.inner(rho, w[i,:])/tal[i, 0]
    ep[i, 1] = np.inner(rho, w[i,:])/tal[i, 1]

sigmaf = np.zeros((N,6))
for i in range(N):
    sigmaf[i,0:3] = (1/(1+(ep[i,0]**2)*(mi[i]**2)))*((mi[i]*np.cross(w[i,:],rho))+(ep[i,0]*(mi[i]**2)*w[i,:])+(tal[i,0]*rho))
    sigmaf[i,3:6] =(1/(1+(ep[i,1]**2)*(mi[i]**2)))*((mi[i]*np.cross(w[i,:],rho))+(ep[i,1]*(mi[i]**2)*w[i,:])+(tal[i,1]*rho))

#Plot dos frames
for i in range(int(sol.y.shape[1])): #t+1
    
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
    
    plt.suptitle('{} individuals - t={}.'.format(N,int(t*i)), size=40)
    
    if i < 10:
        plt.savefig('frames02/0{}.png'.format(i))
    else:
        plt.savefig('frames02/{}.png'.format(i))
    plt.close()