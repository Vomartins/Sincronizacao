import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import broyden1
from scipy.optimize import newton_krylov

np.random.seed(121)
#Modelo de Kuramoto
def kuramoto(t, y, K, N, A, D):

    dydt = np.zeros(3*N)

    for i in range(N):
        a = np.array([y[i], y[i+N], y[i+2*N]],dtype=np.float128)
        s = 0
        for j in range(N):
            b = np.array([y[j], y[j+N], y[j+2*N]],dtype=np.float128)
            s = s + A[i,j]*(b - np.inner(b,a)*a)
        s = (K/D[i])*s

        dydt[i] = s[0]
        dydt[i+N] = s[1]
        dydt[i+2*N] = s[2]

    return dydt

#Parametros
N = int(input("N= "))
ti = int(input("ti= "))
tf = int(input("tf= "))
K = float(input("K= "))
s = [ti, tf]
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
sol = solve_ivp(lambda t, y: kuramoto(t, y, K, N, A, D), s, init_state)

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
    rho[:,i] = (1/D[i])*rho[:,i]
Rho = np.zeros(N)
for i in range(N):
    Rho[i] = np.linalg.norm(rho[:,i])
#print(Rho)
R2 = 0
for i in range(N):
    R2 = R2 + D[i]*Rho[i]
R2 = (1/(N*D_avg))*R2
'''
#Pontos fixos
pontos_fixos = newton_krylov(lambda y: kuramoto(s, y, K, N, A, D),chute_inicial)

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
        +'\nN={}  P={} \nK={} \n'.format(N,p,K), 
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
        plt.savefig('frames_sos/00{}.png'.format(int(i)))
    elif 9 < i < 100:
        plt.savefig('frames_sos/0{}.png'.format(i))
    else:
        plt.savefig('frames_sos/{}.png'.format(int(i)))
    plt.close()
