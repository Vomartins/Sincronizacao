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
        s = s + (a - np.inner(a,a)*a)
        s = (K/N)*s

        W = np.cross(np.transpose(w[:,i]),a)

        dydt[i] = s[0] + n_W[i]*W[0]
        dydt[i+N] = s[1] + n_W[i]*W[1]
        dydt[i+2*N] = s[2] + n_W[i]*W[2]

    return dydt

#Parametros
N = int(input("N= "))
ti = int(input("ti= "))
tf = int(input("tf= "))
K = float(input("K= "))
s = [ti, tf]
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
print(D_avg)

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
E_w = (1/N)*E_w

for i in range(N):
    w[:,i] = w[:,i] - E_w

#Condicao inicial
theta = np.random.uniform(0, 2*np.pi, N)
phi = np.random.uniform(0, np.pi, N)

x0 = np.cos(theta)*np.sin(phi)
y0 = np.sin(theta)*np.sin(phi)
z0 = np.cos(phi)

init_state = np.append(x0, [y0 , z0])

#Chutes inciais
eta = np.random.uniform(0, 2*np.pi, N)
psi = np.random.uniform(0, np.pi, N)

x0_p = np.cos(eta)*np.sin(psi)
y0_p = np.sin(eta)*np.sin(psi)
z0_p = np.cos(psi)

chute_inicial = np.append(x0_p, [y0_p , z0_p])

#Solucao do modelo
sol = solve_ivp(lambda t, y: kuramoto(t, y, K, N, A, D, n_W, w), s, init_state)

#Parametros de ordem
R1 = np.zeros(3)
for i in range(N):
    a = np.array([sol.y[i,-1],sol.y[i+N,-1],sol.y[i+2*N,-1]])
    R1 = R1 + a
R1 = (1/N)*R1
n_R1 = np.linalg.norm(R1)

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

#Pontos fixos
'''
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

d = np.zeros(N)
for i in range(N):
    d[i] = (x_[i]**2 + y_[i]**2 + z_[i]**2)**(1/2)

n_pontos_fixos = np.min(np.abs(d))

print("min(sigma_f): ", np.round(n_pontos_fixos,4))
'''
print("Rho1: ", np.round(n_R1,4))
print("Rho2: ", np.round(R2,4))
print((2*G.number_of_edges())/(N*D_avg))
print((N-1)*p)