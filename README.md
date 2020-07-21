# Sincronização
Simulação e análise de sistemas dinâmicos e sincronização.

1)main.py: Executa o programa kuramoto_rede.py, faz a animação, exclui os frames e move a animação para a pasta onde está.

2)kuramoto_rede.py: Algoritmo de simulação do Modelo de Kuramoto associado a uma rede. O programa consiste em resolver o modelo, calcular os pontos fixos (scipy.optimize.broyden1), classificar os pontos fixos e criar frames para simulaão do modelo (utilizando a equação (2), artigo Ott).

3)Algoritmo de simulação do Modelo de Kuramoto associado a uma rede com uma constante de acoplamento para cada dimensão. O programa consiste em resolver o modelo, calcular os pontos fixos (scipy.optimize.broyden1), classificar os pontos fixos e criar frames para simulaão do modelo (utilizando a equação (2), artigo Ott).
