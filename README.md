# Sincronização
Simulação e análise de sistemas dinâmicos e sincronização.

1)main.py: Executa o programa kuramoto_rede.py, faz a animação, exclui os frames e move a animação para a pasta onde está.

2)kuramoto_rede.py: Algoritmo de simulação do Modelo de Kuramoto associado a uma rede. O programa consiste em resolver o modelo, calcular os pontos fixos (scipy.optimize.broyden1), classificar os pontos fixos e criar frames para visualização do modelo (utilizando a equação (2), artigo Ott).

3)kuramoto01.py: Algoritmo de simulação do Modelo de Kuramoto. O programa consiste em resolver o modelo, calcular os pontos fixos (scipy.optimize.broyden1), classificar os pontos fixos e criar frames para visualização do modelo (utilizando a equação (7), artigo Ott).

4)kuramoto02.py: Algoritmo de simulação do Modelo de Kuramoto. O programa consiste em resolver o modelo, calcular os pontos fixos (scipy.optimize.broyden1), classificar os pontos fixos e criar frames para visualização do modelo (utilizando a equação (2), artigo Ott).

5)pontos_fixos01.py: Calculo dos pontos fixos (scipy.optimize.broyden1; utilizando a equação (7), artigo Ott).

6)pontos_fixos02.py: Calculo dos pontos fixos (scipy.optimize.broyden1; utilizando a equação (2), artigo Ott).