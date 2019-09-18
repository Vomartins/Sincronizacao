import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
#Criar pastas chamadas frames01 e frames02 onde o programa estiver salvo para as figuras

#Modelo de Kuramoto
def kuramoto01(t, y, K, N, A, W):
    sigma = y
    
    d = np.zeros(N)
    for i in range(N):
        for j in range(N):
            d[i] = d[i] + A[i][j] 
    
    dydt = np.zeros(3*N)    
    
    for i in range(N):
        #w = np.random.normal(0, 0.1)
        w = W[i]
        rho = np.zeros(3)
        for j in range(N):
            rho = rho + np.array([A[i][j]*sigma[j], A[i][j]*sigma[j+N], A[i][j]*sigma[j+2*N]])
        rho = (1/N)*rho
        #rho = (1/d[j])*rho
        
        dydt[i] = K*(rho[0] - (np.dot(rho,np.array([sigma[i], sigma[i+N], sigma[i+2*N]])))*sigma[i]) + w*(sigma[i+2*N] - sigma[i+N])

        dydt[i+N] = K*(rho[1] - (np.dot(rho,np.array([sigma[i], sigma[i+N], sigma[i+2*N]])))*sigma[i+N]) + w*(sigma[i] - sigma[i+2*N])

        dydt[i+2*N] = K*(rho[2] - (np.dot(rho,np.array([sigma[i], sigma[i+N], sigma[i+2*N]])))*sigma[i+2*N]) + w*(sigma[i+N] - sigma[i])
        
    return dydt

def kuramoto02(t, y, K, N, A, W):
    sigma = y
    
    d = np.zeros(N)
    for i in range(N):
        for j in range(N):
            d[i] = d[i] + A[i][j] 
    
    dydt = np.zeros(3*N)    
    
    for i in range(N):
        #w = np.random.normal(0, 0.1)
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
s = [0, 300]
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

#Solução com kuramoto01
sol1 = solve_ivp(lambda t, y: kuramoto01(t, y, K, N, A, W), s, init_state)

#Solução com kuramoto02
sol2 =solve_ivp(lambda t, y: kuramoto02(t, y, K, N, A, W), s, init_state)

for i in range(int(sol1.y.shape[1])): #t+1
    #Construção do vetor rho e dos vetores ponto fixo
    rho = np.zeros(3)
    for k in range(N):
        rho = rho + np.array([sol1.y[k,i],sol1.y[k+N,i],sol1.y[k+2*N,i]])

    nrho = np.linalg.norm(rho)
    rho = rho/nrho

    mi = np.zeros(N)
    for k in range(N):
        mi[k] = W[k]/(K*nrho)

    w = np.zeros((N, 3))
    for k in range(N):
        w[k,:] = np.array([W[k], W[k], W[k]])/np.linalg.norm(np.array([W[k], W[k], W[k]]))

    tal = np.zeros((N,2))
    for k in range(N):    
        tal[k, 0] = + (((1 - mi[k]**2)+((mi[k]**2 - 1)**2+4*(mi[k]**2)*(np.dot(rho, w[k,:]))**2)**(1/2))/2)**(1/2)
        tal[k, 1] = - (((1 - mi[k]**2)+((mi[k]**2 - 1)**2+4*(mi[k]**2)*(np.dot(rho, w[k,:]))**2)**(1/2))/2)**(1/2)

    ep = np.zeros((N, 2))
    for k in range(N):
        ep[k, 0] = np.inner(rho, w[k,:])/tal[k, 0]
        ep[k, 1] = np.inner(rho, w[k,:])/tal[k, 1]

    sigmaf = np.zeros((N,6))
    for k in range(N):
        sigmaf[k,0:3] = (1/(1+(ep[k,0]**2)*(mi[k]**2)))*((mi[k]*np.cross(w[k,:],rho))+(ep[k,0]*(mi[k]**2)*w[k,:])+(tal[k,0]*rho))
        sigmaf[k,3:6] =(1/(1+(ep[k,1]**2)*(mi[k]**2)))*((mi[k]*np.cross(w[k,:],rho))+(ep[k,1]*(mi[k]**2)*w[k,:])+(tal[k,1]*rho))
    
    #Plot dos frames
    
    x = sol1.y[0:N,i]
    y = sol1.y[N:2*N,i]
    z = sol1.y[2*N:3*N,i]

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
    
    plt.suptitle('{} individuals - t={}.'.format(N,np.round(sol1.t[i],2)), size=40)
    
    if i < 10:
        plt.savefig('frames01/00{}.png'.format(i))
    elif 9 < i < 100:
        plt.savefig('frames01/0{}.png'.format(i))
    else:
        plt.savefig('frames01/{}.png'.format(int(i)))
    plt.close()
    
for i in range(int(sol2.y.shape[1])): #t+1
    
    #Construção do vetor rho e dos vetores ponto fixo
    rho = np.zeros(3)
    for k in range(N):
        rho = rho + np.array([sol2.y[k,i],sol2.y[k+N,i],sol2.y[k+2*N,i]])

    nrho = np.linalg.norm(rho)
    rho = rho/nrho

    mi = np.zeros(N)
    for k in range(N):
        mi[k] = W[k]/(K*nrho)

    w = np.zeros((N, 3))
    for k in range(N):
        w[k,:] = np.array([W[k], W[k], W[k]])/np.linalg.norm(np.array([W[k], W[k], W[k]]))

    tal = np.zeros((N,2))
    for k in range(N):    
        tal[k, 0] = + (((1 - mi[k]**2)+((mi[k]**2 - 1)**2+4*(mi[k]**2)*(np.dot(rho, w[k,:]))**2)**(1/2))/2)**(1/2)
        tal[k, 1] = - (((1 - mi[k]**2)+((mi[k]**2 - 1)**2+4*(mi[k]**2)*(np.dot(rho, w[k,:]))**2)**(1/2))/2)**(1/2)

    ep = np.zeros((N, 2))
    for k in range(N):
        ep[k, 0] = np.inner(rho, w[k,:])/tal[k, 0]
        ep[k, 1] = np.inner(rho, w[k,:])/tal[k, 1]

    sigmaf = np.zeros((N,6))
    for k in range(N):
        sigmaf[k,0:3] = (1/(1+(ep[k,0]**2)*(mi[k]**2)))*((mi[k]*np.cross(w[k,:],rho))+(ep[k,0]*(mi[k]**2)*w[k,:])+(tal[k,0]*rho))
        sigmaf[k,3:6] =(1/(1+(ep[k,1]**2)*(mi[k]**2)))*((mi[k]*np.cross(w[k,:],rho))+(ep[k,1]*(mi[k]**2)*w[k,:])+(tal[k,1]*rho))
    
    #Plot dos frames
    x = sol2.y[0:N,i]
    y = sol2.y[N:2*N,i]
    z = sol2.y[2*N:3*N,i]

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
    
    plt.suptitle('{} individuals - t={}.'.format(N,np.round(sol2.t[i],2)), size=40)
    
    if i < 10:
        plt.savefig('frames02/00{}.png'.format(i))
    elif 9 < i < 100:
        plt.savefig('frames02/0{}.png'.format(i))
    else:
        plt.savefig('frames02/{}.png'.format(i))
    plt.close()