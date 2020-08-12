import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import broyden1
from scipy.optimize import newton_krylov

np.random.seed(121)
#Modelo de Kuramoto
def kuramoto(t, y, K, N, A, D, n_W, w):

    dydt = np.zeros(3*N)

    W = np.zeros(3)

    for i in range(N):
        a = np.array([y[i], y[i+N], y[i+2*N]])
        s = 0
        for j in range(N):
            b = np.array([y[j], y[j+N], y[j+2*N]])
            s = s + A[i,j]*(b - np.inner(b,a)*a)
        s = (K/N)*s

        W_prod = np.cross(w[:,i].T,a)

        dydt[i] = s[0] + n_W[i]*W_prod[0]
        dydt[i+N] = s[1] + n_W[i]*W_prod[1]
        dydt[i+2*N] = s[2] + n_W[i]*W_prod[2]

    return dydt

#Parametros
N = int(input("N= "))
ti = int(input("ti= "))
tf = int(input("tf= "))
s = [0, tf]
K = float(input("K= "))
mu = float(input("mu= "))
delta = float(input("delta= "))
p = float(input("p= "))
G = nx.gnp_random_graph(N,p)
A = nx.adjacency_matrix(G).A
D = np.zeros(N)
for i in range(N):
    S=0
    for j in range(N):
        S = S+A[i,j]
    D[i] = S
D_avg = 0
for i in range(N):
    D_avg = D_avg + D[i]
D_avg = D_avg/N

#Frequencias naturais
W = np.random.normal(mu, delta, (3,N))
w = np.zeros((3,N))
n_W = np.zeros(N)
for i in range(N):
    
    H = np.zeros((3,3))
    
    H[0,1] = -W[0,i]
    H[0,2] = W[1,i]
    H[1,2] = -W[2,i]
    H[2,1] = W[2,i]
    H[2,0] = -W[1,i]
    H[1,0] = W[0,i]
    
    L, V = np.linalg.eig(H)
    for j in range(3):
        if np.abs(np.imag(L[j]))<(10**(-10)):
            w[:,i] = V[:,j].real.astype(np.float32)
        else:
            n_W[i] = np.abs(np.imag(L[j]))
E_w = np.zeros(3)
for i in range(N):
    E_w = E_w + w[:,i]
E_w = E_w/N
for i in range(N):
    w[:,i] = w[:,i] - E_w

#Condicao inicial
theta = np.random.uniform(0, 2*np.pi, N)
phi = np.random.uniform(0, np.pi, N)

x0 = np.cos(theta)*np.sin(phi)
y0 = np.sin(theta)*np.sin(phi)
z0 = np.cos(phi)

init_state = np.concatenate((x0, y0 , z0))
#Solucao do modelo
sol = solve_ivp(lambda t, y: kuramoto(t, y, K, N, A, D, n_W, w), s, init_state)

print(sol.status)
print(sol.message)
print(sol.success)

#Parametros de ordem
R1 = np.zeros(3)
for i in range(N):
    a = np.array([sol.y[i,-1],sol.y[i+N,-1],sol.y[i+2*N,-1]])
    R1 = R1 + a
R1 = (1/N)*R1

rho = np.zeros((3,N))
for i in range(N):
    for j in range(N):
        a = np.array([sol.y[j,-1],sol.y[j+N,-1],sol.y[j+2*N,-1]])
        rho[:,i] = rho[:,i] + A[i,j]*a
    rho[:,i] = (1/(D[i]))*(rho[:,i])
Rho = np.zeros(N)
for i in range(N):
    Rho[i] = np.linalg.norm(rho[:,i])
#print(Rho)
R2 = 0
for i in range(N):
    R2 = R2 + (D[i])*Rho[i]
R2 = (1/(N*D_avg))*R2
'''
#Pontos fixos
eta = np.random.uniform(0, 2*np.pi, N)
psi = np.random.uniform(0, np.pi, N)

x0_p = np.cos(eta)*np.sin(psi)
y0_p = np.sin(eta)*np.sin(psi)
z0_p = np.cos(psi)

chute_inicial = np.append(x0_p, [y0_p , z0_p])

pontos_fixos = broyden1(lambda y: kuramoto(s, y, K, N, A, D, n_W, w),chute_inicial)

x_i = np.array([])
y_i = np.array([])
z_i = np.array([])
x_e = np.array([])
y_e = np.array([])
z_e = np.array([])

for i in range(N):
    a = np.array([pontos_fixos[i],pontos_fixos[i+N],pontos_fixos[i+2*N]])
    prod_int = np.inner(a,rho[:,i])
    if prod_int < 0 :
        x_i = np.append(x_i, pontos_fixos[i])
        y_i = np.append(y_i, pontos_fixos[i+N])
        z_i = np.append(z_i, pontos_fixos[i+2*N])
    elif prod_int > 0 :
        x_e = np.append(x_e, pontos_fixos[i])
        y_e = np.append(y_e, pontos_fixos[i+N])
        z_e = np.append(z_e, pontos_fixos[i+2*N])

x_ = np.append(x_i, x_e)
y_ = np.append(y_i, y_e)
z_ = np.append(z_i, z_e)

n_pf = np.zeros(N)
for i in range(N):
    n_pf[i] = np.linalg.norm(np.array([x_[i], y_[i], z_[i]]))
print(np.min(np.abs(n_pf)))
'''
for i in range(int(sol.y.shape[1])):

    x = sol.y[0:N,i]
    y = sol.y[N:2*N,i]
    z = sol.y[2*N:3*N,i]

    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 30)

    u, v = np.meshgrid(theta, phi)

    X = np.cos(u)*np.sin(v)
    Y = np.sin(u)*np.sin(v)
    Z = np.cos(v)

    fig = plt.figure(figsize=(6,6))
    ax = fig.gca(projection ='3d',facecolor=(0.8,0.8,1,0.2))
    ax.plot_wireframe(X, Y, Z, color='0.75', alpha='0.3')
    #ax.scatter(x_e,y_e,z_e, c='b', s=15)
    #ax.scatter(x_i,y_i,z_i, c='r', s=15)
    ax.text2D(0.5, 0.03, 
        r'$\rho={} \quad \rho_c={}$'.format(np.round(np.linalg.norm(R1),4),np.round(R2,4))
        +'\nN={}  P={} \nK={} \n'.format(N,p,K)
        +r'$\mu={} \quad \Delta={}$'.format(mu,delta), 
        transform=ax.transAxes, color='black', size='15', ha='center',
        bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))
    plt.axis('off')

    for j in range(N):
        ax1 = fig.gca(projection='3d')
        ax1.scatter(x[j], y[j], z[j], c='k',s=15)

    ax.set_title('{} indivíduos - t={}'.format(N,np.round(sol.t[i],2)), size=25)

    if i < 10:
        fig.savefig('frames_rede/00{}.png'.format(i))
    elif 9 < i < 100:
        fig.savefig('frames_rede/0{}.png'.format(i))
    else:
        fig.savefig('frames_rede/{}.png'.format(i))
    plt.close()
