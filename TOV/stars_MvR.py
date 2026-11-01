import time
import numpy as np
import matplotlib.pyplot as plt
from TOV import TOV

#==================================================================
# Loading data
#==================================================================

path_crust_lg = r"crust_DD2.dat"
path_nuc_n_dd = r"EOS_NUC_T0_dd.txt"
path_nuc_H_nd = r"EOS_HYP_T0_nd.txt"
path_nuc_H_dd = r"EOS_HYP_T0_dd.txt"

ene_dens_crust_lg, pre_crust_lg, num_dens_crust_lg = np.loadtxt(path_crust_lg, unpack=True)# MeV/fm^3, MeV/fm^3, 1/fm^3
ene_dens_nuc_n_dd, pre_nuc_n_dd, num_dens_nuc_n_dd = np.loadtxt(path_nuc_n_dd, unpack=True)# MeV/fm^3, MeV/fm^3, 1/fm^3
ene_dens_nuc_H_nd, pre_nuc_H_nd, num_dens_nuc_H_nd = np.loadtxt(path_nuc_H_nd, unpack=True)# MeV/fm^3, MeV/fm^3, 1/fm^3
ene_dens_nuc_H_dd, pre_nuc_H_dd, num_dens_nuc_H_dd = np.loadtxt(path_nuc_H_dd, unpack=True)# MeV/fm^3, MeV/fm^3, 1/fm^3

ENT0DD = np.append(ene_dens_crust_lg, ene_dens_nuc_n_dd)
EHT0ND = np.append(ene_dens_crust_lg, ene_dens_nuc_H_nd)
EHT0DD = np.append(ene_dens_crust_lg, ene_dens_nuc_H_dd)

PNT0DD = np.append(pre_crust_lg, pre_nuc_n_dd)
PHT0ND = np.append(pre_crust_lg, pre_nuc_H_nd)
PHT0DD = np.append(pre_crust_lg, pre_nuc_H_dd)

NNT0DD = np.append(num_dens_crust_lg, num_dens_nuc_n_dd)
NHT0ND = np.append(num_dens_crust_lg, num_dens_nuc_H_nd)
NHT0DD = np.append(num_dens_crust_lg, num_dens_nuc_H_dd)


ENE = [ENT0DD, EHT0ND, EHT0DD]
PRE = [PNT0DD, PHT0ND, PHT0DD]
NBD = [NNT0DD, NHT0ND, NHT0DD]

number_eos = len(ENE)
colors = ['k', 'r', 'b']#plt.cm.jet(np.linspace(0, 1, number_eos))
labels = ['NUC_EoS_dd','HYP_EoS_nd', 'HYP_EoS_dd']

plt.figure(1)
for press, energ, c, l in zip(PRE, ENE, colors, labels):
    plt.plot(energ, press, c=c, label=l)

plt.ylabel(r'$P [MeV/fm^{3}]$')
plt.xlabel(r'$\varepsilon [MeV/fm^{3}]$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()

#==================================================================
# Create an instance for each EoS
#==================================================================

tov = []
for press, energ, nb_dens in zip(PRE, ENE, NBD):
    tov.append(TOV(energ, press, nb_dens))

#==================================================================
# Computation parameters
#==================================================================

dr = 0.005                      # pure number
dens_c = 1200                   # MeV/fm^3
range_d = np.array([
                   [141, 1430],
                   [141, 1220],
                   [141, 1210]
                   ])              # MeV/fm^3
number_dens = 150

#==================================================================
# Flag for calculation
#==================================================================

one_star = True#False
mass_rad = False
save_MvR = False

#==================================================================
# Solution for one star
#==================================================================

if one_star:

    start = time.time()

    r = []
    P = []
    M = []
    print("===========================================================")
    for t, l in zip(tov, labels):
        print(f'EoS: {l}')
        ri, Pi, Mi, M_grav, prop_mass , bar_mass, B_grav_ene, B_int_ene, tot_ene = t.solve(dens_c, dr)
        r.append(ri)
        P.append(Pi)
        M.append(Mi)
        print(f"Gravitational mass           {M_grav:.3f} M_sun")
        print(f"Proper        mass           {prop_mass:.3f} M_sun")
        print(f"Baryonic      mass           {bar_mass:.3f} M_sun")
        print(f"Gravitational binding energy {B_grav_ene:.3f} x 10^53 ergs")
        print(f"Internal binding energy      {B_int_ene:.3f} x 10^53 ergs")
        print(f"Total binding energy         {tot_ene:.3f} x 10^53 ergs")
        print("===========================================================")


    plt.figure(2, figsize=(10,6))
    plt.suptitle(f'Simulation with central density {dens_c} MeV/fm^{3}', fontsize=15)

    plt.subplot(121)
    plt.title('Mass vs radius')
    for ri, Mi, c, l in zip(r, M, colors, labels):
        plt.plot(ri, Mi, c=c, label=l)

    plt.ylabel('$M/M_{\odot}$')
    plt.xlabel('r [km]')
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(122)
    plt.title('Pressure vs radius')
    for ri, Pi, c, l in zip(r, P, colors, labels):
        plt.plot(ri, Pi, c=c, label=l)
    plt.xlabel('r [km]')
    plt.ylabel(r'P $ \times 10^{36}$ [dyne/cm^2]')
    plt.legend(loc='best')
    plt.grid()

    end = time.time() - start
    print(f"Elapsed time = {end}")

#==================================================================
# Mass radius relation
#==================================================================

if mass_rad :

    start = time.time()
    RDENS = [np.logspace(np.log10(min_d), np.log10(max_d), number_dens) for min_d, max_d in range_d]

    R = []
    M = []
    P = []
    B = []
    BE_G = []
    BE_I = []
    BE_T = []

    for i, dens in enumerate(RDENS):
        Ri, Mi, Pi, Bi, BE_Gi, BE_Ii, BE_Ti = tov[i].MvR(dens, dr)
        R.append(Ri)
        M.append(Mi)
        P.append(Pi)
        B.append(Bi)
        BE_G.append(BE_Gi)
        BE_I.append(BE_Ii)
        BE_T.append(BE_Ti)
        print(f"EoS {i+1} completata")

    plt.figure(3, figsize=(9, 6))
    plt.title('Mass-Radius curve', fontsize=15)
    for ri, Mi, c, l in zip(R, M, colors, labels):
        plt.plot(ri, Mi, c=c, label=l)
    plt.grid()
    plt.ylabel(r'$M/M_{\odot}$')
    plt.xlabel(r'R [km]')
    plt.legend(loc='best')

    lab1 = ["Gravitational Mass", "Proper Mass", "Baryonic Mass"]
    lab2 = ["Gravitational BE", "Internal BE", "Total BE"]
    for j in range(number_eos):
        plt.figure(4+j, figsize=(9, 6))
        plt.suptitle(f'EoS: {labels[j]}')

        plt.subplot(121)
        plt.title("Masses", fontsize=15)
        Q1 = [M[j], P[j], B[j]]
        for i in range(3):
            plt.plot(dens, Q1[i], c=colors[i], label=lab1[i])
        plt.grid()
        plt.xlabel(r'$\varepsilon [MeV/fm^{3}]$')
        plt.ylabel(r'$M/M_{\odot}$')
        plt.legend(loc='best')

        plt.subplot(122)
        plt.title("Binding Energy", fontsize=15)
        Q2 = [BE_G[j], BE_I[j], BE_T[j]]
        for i in range(3):
            plt.plot(dens, Q2[i], c=colors[i], label=lab2[i])
        plt.grid()
        plt.xlabel(r'$\varepsilon [MeV/fm^{3}]$')
        plt.ylabel(r'B x 10^53 [ergs]')
        plt.legend(loc='best')

    end = time.time() - start
    print(f"Elapsed time = {end}")

    #==========================================
    # Maximum mass configuarion
    #==========================================
    print("===========================================================")
    for j in range(number_eos):
        index = 0
        M_max = 0
        for i, m in enumerate(M[j]):
            if m > M_max :
                M_max = m
                index = i
        print(f'Maximum mass configuarion for Eos: {labels[j]}')
        print(f"Gravitational mass           {M_max:.3f} M_sun")
        print(f"Proper        mass           {P[j][index]:.3f} M_sun")
        print(f"Baryonic      mass           {B[j][index]:.3f} M_sun")
        print(f"Gravitational binding energy {BE_G[j][index]:.3f} x 10^53 ergs")
        print(f"Internal binding energy      {BE_I[j][index]:.3f} x 10^53 ergs")
        print(f"Total binding energy         {BE_T[j][index]:.3f} x 10^53 ergs")
        print("===========================================================")

    #==========================================
    # Save data
    #==========================================
    if save_MvR :
        for dens, ri, Mi, Pi, Bi, BE_Gi, BE_Ii, BE_Ti, l in zip(RDENS, R, M, P, B, BE_G, BE_I, BE_T, labels):
            path = fr"/home/francesco/Tesi/codici/TOV/MvR_{l}.txt"
            file = open(path, "w")
            riga_0 = f'#Central energy density [MeV/fm^3] \t'
            riga_1 = f' Mass [M/Msun] \t Radius [km] \t Proper Mass [M/Msun] \t Baryonic Mass [M/Msun] \t'
            riga_2 = f' Gravitational BE [x 10^53 ergs] \t Internal BE [x 10^53 ergs] \t Total BE [x 10^53 ergs] \n'
            file.write(f'{riga_0}{riga_1}{riga_2}')
            for e, m, r, p, b, beg, bei, bet in zip(dens, Mi, ri, Pi, Bi, BE_Gi, BE_Ii, BE_Ti):
                riga = f'{e} \t {m} \t {r} \t {p} \t {b} \t {beg} \t {bei} \t {bet} \n'
                file.write(riga)
            file.close()

plt.show()
