"""
=====================================================================================
This code solves the equations of the relativistic mean field model.
If you want you can consider all coupling constants density-dependent or not
In this code, all calculations are done at finite T.
=====================================================================================
It is possible to choose different physical systems
by setting some flags in the following way:

Nucleonic system:
mat_sim ---> If True the code resolves the equations for symmetric nuclear matter
beta_eq ---> If True the code resolves the equations for asymmetric nuclear matter
             in beta equilibrium between protons, neutrons and electrons
b_eq_mu ---> If True the system is the same of beta_eq but with the addition of muon
b_e_m_n ---> If True the system is the same of b_eq_mu but with the addition of neutrinos

Hyperonic System:
mat_lam ---> If True we considerer protons, neutrons, electrons, muons and Lambda
mat_hyp ---> If True all baryon octet, electrons and muons are put in our mixture
m_hyp_n ---> If True the system is the same of mat_hyp but with the addition of neutrinos

This flag are mutually exclusive,
so if one of them is True all the other must be False
=====================================================================================
By setting the SAVE flag it is also possible to save the data in three files,
one for the abundances and one for the eos and one for chemical potential
=====================================================================================
The code is also set up to decide whether or not
the meson particle interaction is density-dependent:

const_hyp--> If True  the coupling constat for hyperon are constant
             If False the coupling constat for hyperon are a function of density
const_hyp--> If True  the coupling constat for nucleon are constant
             If False the coupling constat for nucleon are a function of density
=====================================================================================
You still need to pay attention to the initial conditions;
the ones set should be reasonable, if necessary change them
=====================================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

from Termo import *
from constant import *
from utilities import *
from mat_sim_system import system_MS
from mat_hyp_system import system_lam, system_hyp, system_all
from mat_asim_system import system_ele, system_muo, system_neu


#==================================================================
# Computation parameters
#==================================================================

number_dens = 200
min_dens    = 0.01
max_dens    = 1

nb_dens = np.logspace(np.log10(min_dens), np.log10(max_dens), number_dens)
#nb_dens = np.linspace(min_dens, max_dens, number_dens)

T = 10/hbc

#==================================================================
# Flag for calculation : ONLY ONE MUST BE TRUE
#==================================================================

mat_sim = False # protons and neutrons
beta_eq = False # protons, neutrons and electrons
b_eq_mu = False # protons, neutrons, electrons and muons
b_e_m_n = False # protons, neutrons, electrons, muons and neutrinos
mat_lam = False # protons, neutrons, electrons, muons and Lambda
mat_hyp = False # baryon octet, electrons and muons
m_hyp_n = True # baryon octet, electrons, muons and neutrinos

#==================================================================
# Path of file for saving data
#==================================================================

SAVE = False# If False the data are not saved
path1 = fr"abb_HYPn_T{int(T*hbc)}_ndd.txt"
path2 = fr"pot_HYPn_T{int(T*hbc)}_ndd.txt"
path3 = fr"EOS_HYPn_T{int(T*hbc)}_ndd.txt"

#==================================================================
# density-dependet coupling-constant
#==================================================================

const_hyp = True  #coupling constant dependent from density or not
const_nuc = False  #coupling constant dependent from density or not

#==================================================================
# Setting species for each system
#==================================================================

if beta_eq or mat_sim:
    muon    = False    # In nuclear matter with beta equlibrium
    Lambda0 = False    # or in symmetric nuclear matter we want
    SX_hyp  = False    # only neutrons, protons and electrons
    neutrin = False    # or only neutron and protons so we set
                       # False all flags for other particles


if b_eq_mu :
    muon    = True     # Now we consider beta equilibrium
    Lambda0 = False    # with also muon and nothing more
    SX_hyp  = False
    neutrin = False

if b_e_m_n :
    muon    = True     # For this we have beta equilibrium
    Lambda0 = False    # considering always muon and also
    SX_hyp  = False    # neutrions
    neutrin = True

if mat_lam :
    muon    = True     # This is for the case with always
    Lambda0 = True     # beta equilibrium and now we have
    SX_hyp  = False    # muon and hyperon lambda only
    neutrin = False    # (generally it is a case that is considered)

if mat_hyp :
    muon    = True     # In this we considerer all particles
    Lambda0 = True     # in barion octet and also muon but
    SX_hyp  = True     # no neutrions
    neutrin = False

if m_hyp_n :
    muon    = True     # All particles
    Lambda0 = True
    SX_hyp  = True
    neutrin = True

#==================================================================
# Start computation
#==================================================================

t0 = time.time()

#array for the solutions
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)
sig_s = np.zeros(number_dens + 1)
phi   = np.zeros(number_dens + 1)
n_pro = np.zeros(number_dens + 1)
n_neu = np.zeros(number_dens + 1)
n_ele = np.zeros(number_dens + 1)
n_muo = np.zeros(number_dens + 1)
n_nuu = np.zeros(number_dens + 1)
n_lam = np.zeros(number_dens + 1)
n_Sgp = np.zeros(number_dens + 1)
n_Sg0 = np.zeros(number_dens + 1)
n_Sgm = np.zeros(number_dens + 1)
n_Xi0 = np.zeros(number_dens + 1)
n_Xim = np.zeros(number_dens + 1)
mu_pr = np.zeros(number_dens + 1)
mu_ne = np.zeros(number_dens + 1)
mu_el = np.zeros(number_dens + 1)
mu_mu = np.zeros(number_dens + 1)
mu_nu = np.zeros(number_dens + 1)
mu_L0 = np.zeros(number_dens + 1)
mu_Sp = np.zeros(number_dens + 1)
mu_S0 = np.zeros(number_dens + 1)
mu_Sm = np.zeros(number_dens + 1)
mu_X0 = np.zeros(number_dens + 1)
mu_Xm = np.zeros(number_dens + 1)



#initial values
sigma[0], omega[0], rho[0] = min_dens * g_sat_NN/m_m[0:3]**2
sig_s[0], phi[0] = min_dens * g_sat_L0[2:4]/m_m[3:5]**2
mu_pr[0], mu_ne[0], mu_L0[0], mu_Sp[0] = m_n[0:4]
mu_S0[0], mu_Sm[0], mu_X0[0], mu_Xm[0] = m_n[4:8]
mu_el[0] = m_l[0]
mu_mu[0] = m_l[1]
mu_nu[0] = m_l[2]
n_pro[0] = 0.01 * min_dens
n_neu[0] = 0.99 * min_dens
n_ele[0] = 0.01 * min_dens
n_muo[0] = 0.00 * min_dens
n_nuu[0] = 0.00 * min_dens
n_lam[0] = 0.00 * min_dens
n_Sgp[0] = 0.00 * min_dens
n_Sg0[0] = 0.00 * min_dens
n_Sgm[0] = 0.00 * min_dens
n_Xi0[0] = 0.00 * min_dens
n_Xim[0] = 0.00 * min_dens


print("Solving system...")
for i, nb in enumerate(nb_dens): #loop over densities

    #constants as the density varies for nucleon
    g_sig_NN = g_sat_NN[0] * f(nb, 0, const=const_nuc)
    g_omg_NN = g_sat_NN[1] * f(nb, 1, const=const_nuc)
    g_rho_NN = g_sat_NN[2] * g(nb,    const=const_nuc)
    #constants as the density varies for Lambda
    g_sig_L0 = g_sat_L0[0] * f(nb, 0, const=const_hyp)
    g_omg_L0 = g_sat_L0[1] * f(nb, 1, const=const_hyp)
    g_phi_L0 = g_sat_L0[2] * f(nb, 1, const=const_hyp)
    g_sgs_L0 = g_sat_L0[3] * f(nb, 0, const=const_hyp)
    #constants as the density varies for Sigma plus
    g_sig_Sp = g_sat_Sp[0] * f(nb, 0, const=const_hyp)
    g_omg_Sp = g_sat_Sp[1] * f(nb, 1, const=const_hyp)
    g_rho_Sp = g_sat_Sp[2] * g(nb,    const=const_hyp)
    #constants as the density varies for Sigma 0
    g_sig_S0 = g_sat_S0[0] * f(nb, 0, const=const_hyp)
    g_omg_S0 = g_sat_S0[1] * f(nb, 1, const=const_hyp)
    #constants as the density varies for Sigma minus
    g_sig_Sm = g_sat_Sm[0] * f(nb, 0, const=const_hyp)
    g_omg_Sm = g_sat_Sm[1] * f(nb, 1, const=const_hyp)
    g_rho_Sm = g_sat_Sm[2] * g(nb,    const=const_hyp)
    #constants as the density varies for Xi 0
    g_sig_X0 = g_sat_X0[0] * f(nb, 0, const=const_hyp)
    g_omg_X0 = g_sat_X0[1] * f(nb, 1, const=const_hyp)
    g_rho_X0 = g_sat_X0[2] * g(nb,    const=const_hyp)
    #constants as the density varies for Xi minus
    g_sig_Xm = g_sat_Xm[0] * f(nb, 0, const=const_hyp)
    g_omg_Xm = g_sat_Xm[1] * f(nb, 1, const=const_hyp)
    g_rho_Xm = g_sat_Xm[2] * g(nb,    const=const_hyp)
    #constants derivatives as the density varies for nucleons
    dg_sig_NN = g_sat_NN[0] * f(nb, 0, const=const_nuc, derive=True)
    dg_omg_NN = g_sat_NN[1] * f(nb, 1, const=const_nuc, derive=True)
    dg_rho_NN = g_sat_NN[2] * g(nb,    const=const_nuc, derive=True)
    #constants derivatives as the density varies for Lambda
    dg_sig_L0 = g_sat_L0[0] * f(nb, 0, const=const_hyp, derive=True)
    dg_omg_L0 = g_sat_L0[1] * f(nb, 1, const=const_hyp, derive=True)
    dg_phi_L0 = g_sat_L0[2] * f(nb, 1, const=const_hyp, derive=True)
    dg_sgs_L0 = g_sat_L0[3] * f(nb, 0, const=const_hyp, derive=True)
    #constants derivatives as the density varies for Sigma plus
    dg_sig_Sp = g_sat_Sp[0] * f(nb, 0, const=const_hyp, derive=True)
    dg_omg_Sp = g_sat_Sp[1] * f(nb, 1, const=const_hyp, derive=True)
    dg_rho_Sp = g_sat_Sp[2] * g(nb,    const=const_hyp, derive=True)
    #constants derivatives as the density varies for Sigma 0
    dg_sig_S0 = g_sat_S0[0] * f(nb, 0, const=const_hyp, derive=True)
    dg_omg_S0 = g_sat_S0[1] * f(nb, 1, const=const_hyp, derive=True)
    #constants derivatives as the density varies for Sigma minus
    dg_sig_Sm = g_sat_Sm[0] * f(nb, 0, const=const_hyp, derive=True)
    dg_omg_Sm = g_sat_Sm[1] * f(nb, 1, const=const_hyp, derive=True)
    dg_rho_Sm = g_sat_Sm[2] * g(nb,    const=const_hyp, derive=True)
    #constants derivatives as the density varies for Xi 0
    dg_sig_X0 = g_sat_X0[0] * f(nb, 0, const=const_hyp, derive=True)
    dg_omg_X0 = g_sat_X0[1] * f(nb, 1, const=const_hyp, derive=True)
    dg_rho_X0 = g_sat_X0[2] * g(nb,    const=const_hyp, derive=True)
    #constants derivatives as the density varies for Xi minus
    dg_sig_Xm = g_sat_Xm[0] * f(nb, 0, const=const_hyp, derive=True)
    dg_omg_Xm = g_sat_Xm[1] * f(nb, 1, const=const_hyp, derive=True)
    dg_rho_Xm = g_sat_Xm[2] * g(nb,    const=const_hyp, derive=True)

    #===============================================================
    # Symmetric matter
    #===============================================================
    if mat_sim :
        #starting point depending on i to speed up the computation
        start = (sigma[i], omega[i], rho[i], mu_pr[i], mu_ne[i])
        #we solve the system and keep the solution
        arg = (nb*0.5, nb*0.5,
               g_sig_NN,  g_omg_NN,  g_rho_NN,
               dg_sig_NN, dg_omg_NN, dg_rho_NN,
               T)

        sol = root(system_MS , start, args=arg, tol=1e-10, method='hybr')
        #save solution
        sigma[i+1], omega[i+1], rho[i+1], mu_pr[i+1], mu_ne[i+1] = sol.x
        n_pro[i+1], n_neu[i+1] = [nb/2]*2

    if not mat_sim :
        #===============================================================
        # Beta equilibrium: p, n, e
        #===============================================================
        #starting point depending on i to speed up the computation
        start = (sigma[i], omega[i], rho[i],
                n_pro[i], n_neu[i], n_ele[i],
                mu_pr[i], mu_ne[i], mu_el[i])
        #we solve the system and keep the solution
        arg = (nb,
               g_sig_NN,  g_omg_NN,  g_rho_NN,
               dg_sig_NN, dg_omg_NN, dg_rho_NN,
               T)
        if beta_eq :
            sol = root(system_ele , start, args=arg, tol=1e-10, method='hybr')
            #save solution
            sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
            n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
            mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]

        #================================================================
        # Beta equilibrium with muon
        #================================================================
        if muon :
            start += (n_muo[i], mu_mu[i])

            if b_eq_mu :
                sol = root(system_muo , start, args=arg, tol=1e-10, method='hybr')
                #save solution
                sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
                n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
                mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]
                n_muo[i+1], mu_mu[i+1]             = sol.x[9:11]

        #================================================================
        # Beta equilibrium with muon and neutrinos
        #================================================================
        if neutrin :
            start += (n_nuu[i], mu_nu[i])

            if b_e_m_n :
                sol = root(system_neu , start, args=arg, tol=1e-10, method='hybr')
                #save solution
                sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
                n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
                mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]
                n_muo[i+1], mu_mu[i+1]             = sol.x[9:11]
                n_nuu[i+1], mu_nu[i+1]             = sol.x[11:13]

        #================================================================
        # Beta equilibrium with muon and Lambda
        #================================================================
        if Lambda0 :
            start += (sig_s[i], phi[i], n_lam[i], mu_L0[i])
            arg   += (g_sig_L0,  g_omg_L0,  g_phi_L0,  g_sgs_L0,
                     dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0)

            if mat_lam :
                sol = root(system_lam , start, args=arg, tol=1e-10, method='hybr')
                #save solution
                sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
                n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
                mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]
                n_muo[i+1], mu_mu[i+1]             = sol.x[9:11]
                sig_s[i+1], phi[i+1]               = sol.x[11:13]
                n_lam[i+1], mu_L0[i+1]             = sol.x[13:15]

        #================================================================
        # All other baryon
        #================================================================
        if SX_hyp:
            start += (n_Sgp[i], n_Sg0[i], n_Sgm[i], mu_Sp[i], mu_S0[i],
                      mu_Sm[i], n_Xi0[i], n_Xim[i], mu_Xm[i], mu_X0[i])
            arg   += (g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
                      dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
                      g_sig_S0,  g_omg_S0,
                      dg_sig_S0, dg_omg_S0,
                      g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
                      dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
                      g_sig_X0,  g_omg_X0,  g_rho_X0,
                      dg_sig_X0, dg_omg_X0, dg_rho_X0,
                      g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
                      dg_sig_Xm, dg_omg_Xm, dg_rho_Xm)

            if mat_hyp :
                sol = root(system_hyp , start, args=arg, tol=1e-10, method='hybr')
                #save solution
                sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
                n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
                mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]
                n_muo[i+1], mu_mu[i+1]             = sol.x[9:11]
                sig_s[i+1], phi[i+1]               = sol.x[11:13]
                n_lam[i+1], mu_L0[i+1]             = sol.x[13:15]
                n_Sgp[i+1], n_Sg0[i+1], n_Sgm[i+1] = sol.x[15:18]
                mu_Sp[i+1], mu_S0[i+1], mu_Sm[i+1] = sol.x[18:21]
                n_Xi0[i+1], n_Xim[i+1]             = sol.x[21:23]
                mu_X0[i+1], mu_Xm[i+1]             = sol.x[23:25]

                # To avoid numerical problem if numerical density became very low
                n_ele[i+1] = 0 if n_ele[i+1]/nb <1e-5 else n_ele[i+1]
                n_muo[i+1] = 0 if n_muo[i+1]/nb <1e-5 else n_muo[i+1]
                n_lam[i+1] = 0 if n_lam[i+1]/nb <1e-5 else n_lam[i+1]
                n_Sgp[i+1] = 0 if n_Sgp[i+1]/nb <1e-5 else n_Sgp[i+1]
                n_Sg0[i+1] = 0 if n_Sg0[i+1]/nb <1e-5 else n_Sg0[i+1]
                n_Sgm[i+1] = 0 if n_Sgm[i+1]/nb <1e-5 else n_Sgm[i+1]
                n_Xi0[i+1] = 0 if n_Xi0[i+1]/nb <1e-5 else n_Xi0[i+1]
                n_Xim[i+1] = 0 if n_Xim[i+1]/nb <1e-5 else n_Xim[i+1]

        #================================================================
        # All particles
        #================================================================
        if m_hyp_n :
            sol = root(system_all , start, args=arg, tol=1e-10, method='hybr')
            #save solution
            sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
            n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
            mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]
            n_muo[i+1], mu_mu[i+1]             = sol.x[9:11]
            n_nuu[i+1], mu_nu[i+1]             = sol.x[11:13]
            sig_s[i+1], phi[i+1]               = sol.x[13:15]
            n_lam[i+1], mu_L0[i+1]             = sol.x[15:17]
            n_Sgp[i+1], n_Sg0[i+1], n_Sgm[i+1] = sol.x[17:20]
            mu_Sp[i+1], mu_S0[i+1], mu_Sm[i+1] = sol.x[20:23]
            n_Xi0[i+1], n_Xim[i+1]             = sol.x[23:25]
            mu_X0[i+1], mu_Xm[i+1]             = sol.x[25:27]

            # To avoid numerical problem if numerical density became very low
            n_ele[i+1] = 0 if n_ele[i+1]/nb <1e-5 else n_ele[i+1]
            n_muo[i+1] = 0 if n_muo[i+1]/nb <1e-5 else n_muo[i+1]
            n_lam[i+1] = 0 if n_lam[i+1]/nb <1e-5 else n_lam[i+1]
            n_Sgp[i+1] = 0 if n_Sgp[i+1]/nb <1e-5 else n_Sgp[i+1]
            n_Sg0[i+1] = 0 if n_Sg0[i+1]/nb <1e-5 else n_Sg0[i+1]
            n_Sgm[i+1] = 0 if n_Sgm[i+1]/nb <1e-5 else n_Sgm[i+1]
            n_Xi0[i+1] = 0 if n_Xi0[i+1]/nb <1e-5 else n_Xi0[i+1]
            n_Xim[i+1] = 0 if n_Xim[i+1]/nb <1e-5 else n_Xim[i+1]

    print(f"{(i+1)/(number_dens) * 100:.1f} % \r", end='')


#we delete the initial value
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]
sig_s = sig_s[1:]
phi   = phi[1:]
n_pro = n_pro[1:]
n_neu = n_neu[1:]
n_ele = n_ele[1:]
n_muo = n_muo[1:]
n_nuu = n_nuu[1:]
n_lam = n_lam[1:]
n_Sgp = n_Sgp[1:]
n_Sg0 = n_Sg0[1:]
n_Sgm = n_Sgm[1:]
n_Xi0 = n_Xi0[1:]
n_Xim = n_Xim[1:]
mu_pr = mu_pr[1:]
mu_ne = mu_ne[1:]
mu_L0 = mu_L0[1:]
mu_Sp = mu_Sp[1:]
mu_S0 = mu_S0[1:]
mu_Sm = mu_Sm[1:]
mu_X0 = mu_X0[1:]
mu_Xm = mu_Xm[1:]
mu_el = mu_el[1:]
mu_mu = mu_mu[1:]
mu_nu = mu_nu[1:]

#==================================================================
# Computing termodinamical quantities
#==================================================================

print("Computing termodinamical quantities...")
#energy and pressure computation
ene_nb = Total_energy(nb_dens, sigma, omega, rho, phi, sig_s,
                      mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                      mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T,
                      mat_sim, muon, Lambda0, SX_hyp, neutrin,
                      const_hyp, const_nuc)
print("Energy done")

pre_nb = Total_pressure(nb_dens, sigma, omega, rho, phi, sig_s,
                        n_pro, n_neu, n_lam, n_Sgp, n_Sg0, n_Sgm,
                        n_Xi0, n_Xim, mu_pr, mu_ne, mu_L0, mu_Sp,
                        mu_S0, mu_Sm, mu_X0, mu_Xm, mu_el, mu_mu,
                        mu_nu, T,
                        mat_sim, muon, Lambda0, SX_hyp, neutrin,
                        const_hyp, const_nuc)
print("Pressure done")

ent_nb = Total_entropy(nb_dens, sigma, omega, rho, phi, sig_s,
                       mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                       mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T,
                       mat_sim, muon, Lambda0, SX_hyp, neutrin,
                       const_hyp, const_nuc)
print("Entropy done")

dt = time.time() - t0
print(f'tempo impeigato: {dt}')

#enrgy per nucleon
E_over_A =  hbc*(ene_nb/nb_dens)
#pressure
pre_nb *= hbc
#entropy
ent_nb /= nb_dens

#==================================================================
# Plot
#==================================================================

#abundances plot
plt.figure(1)
plt.title('Abundance of each species')
plt.xlabel('density [$fm^{-3}$]')
plt.plot(nb_dens, n_pro/nb_dens, 'red',     label='$n_p/n_b$')
plt.plot(nb_dens, n_neu/nb_dens, 'black',   label='$n_n/n_b$')
plt.plot(nb_dens, n_ele/nb_dens, 'blue',    label='$n_e/n_b$')
plt.plot(nb_dens, n_muo/nb_dens, 'purple',  label='$n_{\mu}/n_b$')
plt.plot(nb_dens, n_lam/nb_dens, 'green',   label='$n_{\Lambda}/n_b$')
plt.plot(nb_dens, n_Sgp/nb_dens, 'orange',  label='$n_{\Sigma^+}/n_b$')
plt.plot(nb_dens, n_Sg0/nb_dens, 'lime',    label='$n_{\Sigma^0}/n_b$')
plt.plot(nb_dens, n_Sgm/nb_dens, 'pink',    label='$n_{\Sigma^-}/n_b$')
plt.plot(nb_dens, n_Xi0/nb_dens, 'navy',    label='$n_{\Xi^0}/n_b$')
plt.plot(nb_dens, n_Xim/nb_dens, 'fuchsia', label='$n_{\Xi^-}/n_b$')
plt.plot(nb_dens, n_nuu/nb_dens, 'deepskyblue', label=r'$n_{\nu}/n_b$')
plt.legend(loc='best', ncol=2)
plt.yscale('log')
plt.grid()


#chemical potential plot
plt.figure(2)
plt.title(f'chemical potential for protons and neutrons at T = {int(T*hbc):.0f} MeV')
plt.ylabel('chemical potential [$fm^{-1}$]')
plt.xlabel('density [$fm^{-3}$]')
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
plt.legend(loc='best')


#energy plot
plt.figure(3)
plt.title(f'Binding energy per nucleon at T = {int(T*hbc):.0f} MeV')
plt.ylabel('E/A [Mev]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A, 'k')
#plt.savefig("Energia_T10")

#pressure plot
plt.figure(4)
plt.title(f'Pressure at T = {int(T*hbc):.0f} MeV')
plt.ylabel('P [Mev/$fm^3$]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, pre_nb, 'k')


#pressure plot
plt.figure(5)
plt.title(f'Entropy at T = {int(T*hbc):.0f} MeV')
plt.ylabel('S [Mev/K]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, ent_nb, 'k')


plt.show()

#==================================================================
# Save data
#==================================================================

if SAVE :

    file = open(path1, "w")
    file.write("# nb [fm^-3] \t Yp \t Yn \t Ye \t Ym \t Ynu \t YL \t YSp \t YS0 \t YSm \t YX0 \t YXm \n")

    List = zip(nb_dens, n_pro, n_neu, n_ele, n_muo, n_nuu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim)

    for nb, p, n, e, m, nu, L, Sp, S0, Sm, X0, Xm in List:
        data1 = f"{nb} \t {p/nb} \t {n/nb} \t {e/nb} \t {m/nb} \t {nu/nb} \t {L/nb}"
        data2 = f"{Sp/nb} \t {S0/nb} \t {Sm/nb} \t {X0/nb} \t {Xm/nb} \n"
        file.write(f"{data1} \t {data2}")

    file.close()

    #======================================

    file = open(path2, "w")
    str1 = f'# nb [fm^-3] \t mu_n [fm^-1] \t mu_p [fm^-1] \t mu_e [fm^-1] \t mu_m [fm^-1] \t mu_nu [fm^-1] \t'
    str2 = f'  mu_L0 [fm^-1] \t mu_Sp [fm^-1] \t mu_S0 [fm^-1] \t mu_Sm [fm^-1] \t mu_X0 [fm^-1] \t mu_Xm [fm^-1] \n'
    file.write(f'{str1}{str2}')

    List = zip(nb_dens, mu_ne, mu_pr, mu_el, mu_mu, mu_nu, mu_L0, mu_Sp, mu_S0, mu_Sm, mu_X0, mu_Xm)

    for nb, n, p, e, m, nu, L0, Sp, S0, Sm, X0, Xm in List:
        D = f"{nb} \t {n} \t {p} \t {e} \t {m} \t{nu} \t {L0} \t {Sp} \t {S0} \t {Sm} \t {X0} \t {Xm} \n"
        file.write(D)

    file.close()

    #======================================

    file = open(path3, "w")
    file.write("# E/A [MeV fm^-3] \t P [MeV fm^-3] \t S \t nb [fm^-3] \n")
    for nb, ene, pre, ent in zip(nb_dens, E_over_A, pre_nb, ent_nb):
        file.write(f"{ene} \t{pre} \t {ent} \t {nb} \n")

    file.close()
