import numpy as np
import matplotlib.pyplot as plt

S = 2

path1 = fr"abb_HYPn_S{S}_dd.txt"
path2 = fr"pot_HYPn_S{S}_dd.txt"
path3 = fr"EOS_HYPn_S{S}_dd.txt"
path4 = fr"abb_HYP_S{S}_dd.txt"
path5 = fr"pot_HYP_S{S}_dd.txt"
path6 = fr"EOS_HYP_S{S}_dd.txt"
"""
nb_dens, n_pro, n_neu, n_ele, n_muo, n_nuu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim = np.loadtxt(path1, unpack=True)
plt.figure(1)
plt.title(f'Abundance of each species at S/A = {S}', fontsize=15)
plt.xlabel('$n_B$ [fm$^{-3}$]', fontsize=15)
plt.ylabel('$Y_i$', fontsize=15)
plt.plot(nb_dens, n_pro, 'red',     label='$n_p/n_b$')
plt.plot(nb_dens, n_neu, 'black',   label='$n_n/n_b$')
plt.plot(nb_dens, n_ele, 'blue',    label='$n_e/n_b$')
plt.plot(nb_dens, n_muo, 'purple',  label='$n_{\mu}/n_b$')
plt.plot(nb_dens, n_lam, 'green',   label='$n_{\Lambda}/n_b$')
plt.plot(nb_dens, n_Sgp, 'orange',  label='$n_{\Sigma^+}/n_b$')
plt.plot(nb_dens, n_Sg0, 'lime',    label='$n_{\Sigma^0}/n_b$')
plt.plot(nb_dens, n_Sgm, 'pink',    label='$n_{\Sigma^-}/n_b$')
plt.plot(nb_dens, n_Xi0, 'navy',    label='$n_{\Xi^0}/n_b$')
plt.plot(nb_dens, n_Xim, 'fuchsia', label='$n_{\Xi^-}/n_b$')
plt.plot(nb_dens, n_nuu, 'deepskyblue', label=r'$n_{\nu}/n_b$')
plt.yscale('log')
plt.grid()

nb_dens, n_pro, n_neu, n_ele, n_muo, n_nuu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim = np.loadtxt(path4, unpack=True)


plt.plot(nb_dens, n_pro, 'red',    linestyle='--', label='$n_p/n_b$')
plt.plot(nb_dens, n_neu, 'black',   linestyle='--',label='$n_n/n_b$')
plt.plot(nb_dens, n_ele, 'blue',    linestyle='--',label='$n_e/n_b$')
plt.plot(nb_dens, n_muo, 'purple',  linestyle='--',label='$n_{\mu}/n_b$')
plt.plot(nb_dens, n_lam, 'green',   linestyle='--',label='$n_{\Lambda}/n_b$')
plt.plot(nb_dens, n_Sgp, 'orange',  linestyle='--',label='$n_{\Sigma^+}/n_b$')
plt.plot(nb_dens, n_Sg0, 'lime',    linestyle='--',label='$n_{\Sigma^0}/n_b$')
plt.plot(nb_dens, n_Sgm, 'pink',    linestyle='--',label='$n_{\Sigma^-}/n_b$')
plt.plot(nb_dens, n_Xi0, 'navy',    linestyle='--',label='$n_{\Xi^0}/n_b$')
plt.plot(nb_dens, n_Xim, 'fuchsia', linestyle='--',label='$n_{\Xi^-}/n_b$')
plt.plot(nb_dens, n_nuu, 'deepskyblue', linestyle='--',label=r'$n_{\nu}/n_b$')
plt.legend(loc='best')
#============================================================================================

nb_dens, mu_ne, mu_pr, mu_el, mu_mu, mu_nu, mu_L0, mu_Sp, mu_S0, mu_Sm, mu_X0, mu_Xm = np.loadtxt(path2, unpack=True)
plt.figure(2)
plt.title(f'chemical potential for protons and neutrons at S = {S}', fontsize=15)
plt.ylabel('chemical potential [fm]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, mu_ne, 'k', label='$\mu_n$')
plt.plot(nb_dens, mu_pr, 'red', label='$\mu_p$')
plt.plot(nb_dens, mu_el, 'blue',label='$\mu_e$')
plt.plot(nb_dens, mu_mu, 'purple', label='$\mu_{\mu}$')
plt.plot(nb_dens, mu_L0, 'green',   label='$\mu_{\Lambda}$')
plt.plot(nb_dens, mu_Sp, 'orange',  label='$\mu_{\Sigma^+}$')
plt.plot(nb_dens, mu_S0, 'lime',    label='$\mu_{\Sigma^0}$')
plt.plot(nb_dens, mu_Sm, 'pink',    label='$\mu_{\Sigma^-}$')
plt.plot(nb_dens, mu_X0, 'navy',    label='$\mu_{\Xi^0}$')
plt.plot(nb_dens, mu_Xm, 'fuchsia', label='$\mu_{\Xi^-}$')
plt.plot(nb_dens, mu_nu, 'deepskyblue', label=r'$\mu_{\nu}$')

nb_dens, mu_ne, mu_pr, mu_el, mu_mu, mu_nu, mu_L0, mu_Sp, mu_S0, mu_Sm, mu_X0, mu_Xm = np.loadtxt(path5, unpack=True)

plt.plot(nb_dens, mu_ne, 'k', linestyle='--',label='$\mu_n$')
plt.plot(nb_dens, mu_pr, 'red', linestyle='--',label='$\mu_p$')
plt.plot(nb_dens, mu_el, 'blue',linestyle='--',label='$\mu_e$')
plt.plot(nb_dens, mu_mu, 'purple', linestyle='--',label='$\mu_{\mu}$')
plt.plot(nb_dens, mu_L0, 'green',  linestyle='--', label='$\mu_{\Lambda}$')
plt.plot(nb_dens, mu_Sp, 'orange', linestyle='--', label='$\mu_{\Sigma^+}$')
plt.plot(nb_dens, mu_S0, 'lime',   linestyle='--', label='$\mu_{\Sigma^0}$')
plt.plot(nb_dens, mu_Sm, 'pink',   linestyle='--', label='$\mu_{\Sigma^-}$')
plt.plot(nb_dens, mu_X0, 'navy',   linestyle='--', label='$\mu_{\Xi^0}$')
plt.plot(nb_dens, mu_Xm, 'fuchsia', linestyle='--',label='$\mu_{\Xi^-}$')
plt.plot(nb_dens, mu_nu, 'deepskyblue', linestyle='--',label=r'$\mu_{\nu}$')
plt.legend(loc='best')
"""
#=================================================================================================

ene_nb, pre_nb, T_nbd, nb_dens = np.loadtxt(path3, unpack=True)
ene_nb1, pre_nb1, T_nbd1, nb_dens1 = np.loadtxt(path6, unpack=True)

plt.figure(3)
plt.title(f'Binding energy per nucleon at S = {S}', fontsize=15)
plt.ylabel('E/A [Mev]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, ene_nb, 'k')
plt.plot(nb_dens1, ene_nb1, 'k--')
#plt.savefig("Energia_S3")

#pressure plot
path32 = fr"EOS_HYPn_S{S}_dd.txt"
ene_nb2, pre_nb2, T_nbd2, nb_dens2 = np.loadtxt(path32, unpack=True)

plt.figure(4)
plt.title(f'Pressure at S = {S}', fontsize=15)
plt.ylabel('P [Mev/$fm^3$]', fontsize=15)
plt.xlabel('energy density [MeV fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(ene_nb, pre_nb, 'k')
plt.plot(ene_nb1, pre_nb1, 'r--')
#plt.plot(ene_nb2, pre_nb2, 'b:')
#plt.savefig("Pressione_S3")

#Temperature plot
plt.figure(5)
plt.title(f'Temperature at S/A = {S}', fontsize=15)
plt.ylabel('T [K]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, T_nbd, 'k')
plt.plot(nb_dens1, T_nbd1, 'k--')


plt.show()
