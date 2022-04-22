from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import math


#DIMENSÃO DO SISTEMA#
N = 5  #Dimensão do espaço de Hilbert da cavidade
idatomA = qeye(3) #Operador identidade do átomo A
ida = qeye(N) #Operador identidade do campo
nloop = 1000 #Número de passos em DeltaP

#PARÂMETROS DO HAMILTONIANO#
g = 50.0 #Acoplamento átomo-campo (transição atômica 1<->3
E = 1.0#Campo de prova
O = 40.0 #Frequência de Rabi do campo de controle
D1 = 0.0 #Dessintonia entre átomo e cavidade (w0-wp)
D2 = 0.0 #Dessintonia entre átomo e campo de controle (w0-wc)
DP = 100.0 #Variação de DP -> Dessintonia entre modo da cavidade e campo de prova
DPList= np.linspace(-DP,DP,nloop) #Divisão dos passos

#PARÂMETROS DA EQUAÇÃO MESTRA - Taxa de decaimento e dissipação#
Gamma31 = 0.1 #Taxa de decaimento 3 -> 1
Gamma32 = 0.1 #Taxa de decaimento 3 -> 2
gamma2 = 0.0 #Taxa de defasagem - level 2
gamma3 = 0.0 #Taxa de defasagem - level 3
kappa =1.0 #Decaimento da cavidade

#OPERADORES ATÔMICOS#
s12=Qobj([[0,1,0],[0,0,0],[0,0,0]])
s13=Qobj([[0,0,1],[0,0,0],[0,0,0]])
s23=Qobj([[0,0,0],[0,0,1],[0,0,0]])

#Átomo A#
S13A = tensor(ida,s13) #sigma13
S23A = tensor(ida,s23) #sigma23
S11A = S13A*S13A.dag() #sigma11
S22A = S23A*S23A.dag()#sigma22
S33A = S23A.dag()*S23A #sigma33

#OPERADORES DO MODO DA CAVIDADE#
a=tensor(destroy(N),idatomA)


# Colapse Operators
C1 = math.sqrt(2*kappa)*a      #cavity mode 
C31 = math.sqrt(2*Gamma31)*S13A  #decaimento atom 31
C32 = math.sqrt(2*Gamma32)*S23A  #decaient0 atom 32
C22 = math.sqrt(2*gamma2)*S22A  #defasagem atom 22
C33 = math.sqrt(2*gamma3)*S33A  #defasagem atom 33

C_list = [C1, C31, C32, C22, C33]

#HAMILTONIANO#
H1 = O*S23A + O*S23A.dag()+D1*(S33A) + D1*S22A - D2*(S22A) + g*S13A.dag()*a + g*a.dag()*S13A +E*a+E*a.dag() #Interação 

#SIMULAÇÃO#
C=[]
for k in range(0,nloop):
    DP=DPList[k] #passos
    H= DP*S11A - DP*(a.dag()*a)+H1 #Hamiltoniano total
    rhoss=steadystate(H, C_list) #Retorna a matriz densidade representando o estado estacionário do Liouviliano (eu espero)
    Num=expect(a.dag()*a.dag()*a*a, rhoss)
    Den= expect(a.dag()*a, rhoss)#Transmissão normalizada
    Cor=Num/(Den**2)
    Cor= Cor.real
    C+=[Cor]
    #if k%10==0:
        #print(k/10,'%')
        #print(np.array(OList[0:k+1]))
        #print(C)
        #f = open("1atomo-dados-backup", "w")
        #f.write("x y\n")        # column names
        #np.savetxt(f, np.array([OList[0:k+1], C]).T)


#PLOT#
plt.figure(1, dpi=300)
plt.yscale("log")
plt.plot(DPList,C, linewidth=2 )
plt.savefig('figure.png', format='png')
