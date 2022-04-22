from qutip import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#DIMENSÃO DO SISTEMA#
N = 5  #Dimensão do espaço de Hilbert da cavidade
idatomA = qeye(3) #Operador identidade do átomo A
ida = qeye(N) #Operador identidade do campo
nloop = 500 #Número de passos em DeltaP

#PARÂMETROS DO HAMILTONIANO#
g = 50.0 #Acoplamento átomo-campo (transição atômica 1<->3
E = 1.0#Campo de prova
O = 250.0 #Frequência de Rabi do campo de controle
D1 = 0.0 #Dessintonia entre átomo e cavidade (w0-wp)
D2 = 0.0 #Dessintonia entre átomo e campo de controle (w0-wc)
DP = 100.0 #Variação de DP -> Dessintonia entre modo da cavidade e campo de prova
DPList= np.linspace(-DP,DP,nloop) #Divisão dos passos
OList= np.linspace(0,O,nloop)

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

#EQUAÇÃO MESTRA = ÁTOMO DE 2 NÍVEIS + MODO DA CAVIDADE#

# Colapse Operators
C1 = math.sqrt(2*kappa)*a      #cavity mode 
C31 = math.sqrt(2*Gamma31)*S13A  #decaimento atom 31
C32 = math.sqrt(2*Gamma32)*S23A  #decaient0 atom 32
C22 = math.sqrt(2*gamma2)*S22A  #defasagem atom 22
C33 = math.sqrt(2*gamma3)*S33A  #defasagem atom 33

C_list = [C1, C31, C32, C22, C33]

#HAMILTONIANO#
H1 = D1*(S33A) + D1*S22A - D2*(S22A) + g*S13A.dag()*a + g*a.dag()*S13A +E*a+E*a.dag() #Interação 

#SIMULAÇÃO#
def correl(DPList,OList):
    select=np.array(qeye(nloop))
    for k in range(0,nloop):
        for l in range (0,nloop):
            Dp=DPList[k,l]
            Oc=OList[k,l]
            H = Oc*S23A + Oc*S23A.dag() + Dp*S11A - Dp*a.dag()*a + H1 
            rhoss=steadystate(H, C_list, method='eigen') #Retorna a matriz densidade representando o estado estacionário do Liouviliano (eu espero)
            Num=expect(a.dag()*a.dag()*a*a, rhoss)
            Den= expect(a.dag()*a, rhoss)#Transmissão normalizada
            Cor=Num/(Den**2)
            Cor= Cor.real
            select[k,l]=Cor
        if k%10==0:
            print(k/10,'%')
    return select


#PLOT#

#plt.plot(DPList,C, linewidth=2 )

DPList, OList = np.meshgrid(DPList, OList)
C = correl(DPList,OList)
C = C.real

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

# Plot the surface.
plt.figure(1, dpi=300)
surf = ax.plot_surface(OList, DPList, np.log10(np.array(C)),rstride=1,cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True) #gráfico de superfície
#surf = ax.plot_wireframe(OList, DPList, np.log10(np.array(C)),rstride=50,cstride=50, linewidth=1) #gráfico de linhas

fig.colorbar(surf, fraction=0.10, shrink=0.5, pad=0, panchor=(1,1)) #colorbar

plt.show()
