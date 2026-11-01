"""
We want to find at which T the pressure has an inflection
with horizontal tangent then we compute the derivative and
we find all the minima of the derivative. After this we find
the zeros of the minima of the derivative as a function of T
because the derivative of a function that has inflection
with horizontal tangent must have minimum equal to zero
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

path  = fr"data_pt"
path1 = fr"data_pt/parameters.txt"
#path0 = fr"EOS_SNM_T0_dd.txt"

#ene, pre0, nb0 = np.loadtxt(path0, unpack=True)
par = np.loadtxt(path1, unpack=True)
number_dens, min_dens, max_dens = par[0:3]
number_dens = int(number_dens)
Ture = par[3:]
Tmin = np.min(Ture)
Tmax = np.max(Ture)
Tnum = len(Ture)

nb = np.linspace(min_dens, max_dens, number_dens) #array of numerical density
P  = np.zeros((Tnum, number_dens))                #matrix for pressure
dP = np.zeros((Tnum, number_dens-2))              #matrix for presuure derivative


for i in range(Tnum):
    nome = f'/dati{i}.npy'
    P[i, :] = np.load(path+nome, allow_pickle=True)

for i in range(Tnum):
    for j in range(1, number_dens-1):
        dP[i, j-1] = (P[i, j+1] - P[i, j-1])/(nb[j+1] - nb[j-1])

#find all minima and intepolate
min_der = np.array([np.min(dP[i, :]) for i in range(Tnum)])
sp_min = interp1d(Ture, min_der, kind='cubic')
t = np.linspace(Tmin, Tmax, 100000)
#find the zeros
sol = root_scalar(sp_min, method='bisect', bracket=[Tmin, Tmax])
T_c = sol.root
print(f'T_c = {T_c:.5f} MeV')

#======================================================================================
# PLOT PRESSURE
#======================================================================================

colors = plt.cm.jet(np.linspace(0, 1, Tnum))
##plot
plt.figure(1, figsize=(7, 7))
plt.title('Pressure', fontsize=15)
plt.ylabel('P [MeV/fm$^3$]', fontsize=15)
plt.xlabel('$n_B$ [fm$^{-3}$]', fontsize=15)
#plt.grid()
plt.minorticks_on()
#plt.plot(nb0, pre0, 'k', label='T = 0 MeV')
plt.xlim(0, 0.19)
plt.ylim(-0.8, 1.1)

for i in range(Tnum):
    plt.plot(nb, P[i, :], color=colors[i], label=f'T = {Ture[i]:.3f} MeV')

plt.legend(loc='best')#, ncol=2)
#plt.savefig('/home/francesco/GitHub/tesi/IMMAGINI/P_SNM_T.pdf')

#======================================================================================
# PLOT DERIVATIVE
#======================================================================================

fig = plt.figure(2, figsize=(18,10))

main_ax = fig.add_subplot()
main_ax.set_title('Pressure derivative', fontsize=25)
main_ax.set_xlabel('$n_B$ [fm$^{-3}$]', fontsize=25)
main_ax.set_ylabel('dP/dn [MeV]', fontsize=25)
main_ax.set_ylim(-1, 4)
main_ax.set_xlim(0.02, 0.12)
#main_ax.grid()
main_ax.minorticks_on()

for i in range(Tnum):
    main_ax.plot(nb[1:-1], dP[i, :], color=colors[i], label=f'T = {Ture[i]:.3f} MeV')

main_ax.legend(loc='best')

right_inset_ax = fig.add_axes([.6, .17, .25, .25])
#right_inset_ax.grid()
right_inset_ax.set_title('minimum of pressure derivative', fontsize=14)
right_inset_ax.set_ylabel('min(dP/dn) [MeV]', fontsize=14)
right_inset_ax.set_xlabel('temperature [MeV]', fontsize=14)
right_inset_ax.minorticks_on()

right_inset_ax.plot(Ture, min_der, 'b')
#right_inset_ax.plot(t, sp_min(t),'r')
#plt.savefig('transition.pdf')
plt.show()
