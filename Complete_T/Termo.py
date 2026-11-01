"""
Code with function for computing thermodynamics quantities
"""

import numpy as np

from constant import *
from utilities import *


def Total_energy(nb_dens, sigma, omega, rho, phi, sig_s,
                 mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                 mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T,
                 mat_sim, muon, Lambda0, SX_hyp, neutrin,
                 const_hyp, const_nuc):
    '''
    Computation of the system's total energy

    Parameters
    ----------
    T : float
        temperature

    Retunrs
    -------
    energ : 1darray
        array for the total energy
    '''
    number_dens = len(nb_dens)
    #array for the energy
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):

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

        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sig_NN
        m_eff_n = m_n[1] - sigma[i]*g_sig_NN
        #energy terms
        Sigma_Np = g_omg_NN*omega[i] - g_rho_NN*rho[i]
        Sigma_Nn = g_omg_NN*omega[i] + g_rho_NN*rho[i]
        #rearrangemet terms
        Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omega[i])**2
        Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho[i]  )**2
        Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sigma[i])**2
        #nucleons energy
        ene_p  = Inte(ene, (m_eff_p, Sigma_Np + Sigma_r_NN, mu_pr[i], T))
        ene_n  = Inte(ene, (m_eff_n, Sigma_Nn + Sigma_r_NN, mu_ne[i], T))
        #Total baryon energy
        tot_ene = ene_p + ene_n

        if not mat_sim:
            tot_ene += Inte(ene, (m_l[0], 0, mu_el[i], T))

            if muon:
                tot_ene += Inte(ene, (m_l[1], 0, mu_mu[i], T))

            if neutrin :
                tot_ene += Inte(ene, (m_l[2], 0, mu_nu[i], T))

            if Lambda0 :
                m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0 - sig_s[i]*g_sgs_L0
                Sigma_L0 = g_omg_L0*omega[i] + g_phi_L0*phi[i]
                Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omega[i])**2
                Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sigma[i])**2
                Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sig_s[i])**2
                Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi[i]  )**2
                tot_ene += Inte(ene, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0[i], T))

            if SX_hyp :
                m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp
                Sigma_Sp = g_omg_Sp*omega[i] - g_rho_Sp*rho[i]
                Sigma_r_Sp  = (dg_omg_Sp/g_omg_Sp)*(m_m[1]*omega[i])**2
                Sigma_r_Sp += (dg_rho_Sp/g_rho_Sp)*(m_m[2]*rho[i]  )**2
                Sigma_r_Sp -= (dg_sig_Sp/g_sig_Sp)*(m_m[0]*sigma[i])**2
                tot_ene += Inte(ene, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp[i], T))
                #
                m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0
                Sigma_S0 = g_omg_S0*omega[i]
                Sigma_r_S0  = (dg_omg_S0/g_omg_S0)*(m_m[1]*omega[i])**2
                Sigma_r_S0 -= (dg_sig_S0/g_sig_S0)*(m_m[0]*sigma[i])**2
                tot_ene += Inte(ene, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0[i], T))
                #
                m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm
                Sigma_Sm = g_omg_Sm*omega[i] + g_rho_Sm*rho[i]
                Sigma_r_Sm  = (dg_omg_Sm/g_omg_Sm)*(m_m[1]*omega[i])**2
                Sigma_r_Sm += (dg_rho_Sm/g_rho_Sm)*(m_m[2]*rho[i]  )**2
                Sigma_r_Sm -= (dg_sig_Sm/g_sig_Sm)*(m_m[0]*sigma[i])**2
                tot_ene += Inte(ene, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm[i], T))
                #
                m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0
                Sigma_X0 = g_omg_X0*omega[i] - g_rho_X0*rho[i]
                Sigma_r_X0  = (dg_omg_X0/g_omg_X0)*(m_m[1]*omega[i])**2
                Sigma_r_X0 += (dg_rho_X0/g_rho_X0)*(m_m[2]*rho[i]  )**2
                Sigma_r_X0 -= (dg_sig_X0/g_sig_X0)*(m_m[0]*sigma[i])**2
                tot_ene += Inte(ene, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0[i], T))
                #
                m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm
                Sigma_Xm = g_omg_Xm*omega[i] + g_rho_Xm*rho[i]
                Sigma_r_Xm  = (dg_omg_Xm/g_omg_Xm)*(m_m[1]*omega[i])**2
                Sigma_r_Xm += (dg_rho_Xm/g_rho_Xm)*(m_m[2]*rho[i]  )**2
                Sigma_r_Xm -= (dg_sig_Xm/g_sig_Xm)*(m_m[0]*sigma[i])**2
                tot_ene += Inte(ene, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm[i], T))

        #mesons energy
        ene_m  = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        ene_m += 0.5*(m_m[3]**2*sig_s[i]**2 + m_m[4]**2*phi[i]**2)
        tot_ene += ene_m

        #system's total energy
        energ[i] =  tot_ene

    return energ


def Total_pressure(nb_dens, sigma, omega, rho, phi, sig_s,
                   n_pro, n_neu, n_lam, n_Sgp, n_Sg0, n_Sgm,
                   n_Xi0, n_Xim, mu_pr, mu_ne, mu_L0, mu_Sp,
                   mu_S0, mu_Sm, mu_X0, mu_Xm, mu_el, mu_mu,
                   mu_nu, T,
                   mat_sim, muon, Lambda0, SX_hyp, neutrin,
                   const_hyp, const_nuc):
    '''
    Computation of the system's total pressure

    Parameters
    ----------
    T : float
        temperature

    Retunrs
    -------
    press : 1darray
        array for the total pressure
    '''
    number_dens = len(nb_dens)
    #array for the pressure
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):

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

        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sig_NN
        m_eff_n = m_n[1] - sigma[i]*g_sig_NN
        #energy terms
        Sigma_Np = g_omg_NN*omega[i] - g_rho_NN*rho[i]
        Sigma_Nn = g_omg_NN*omega[i] + g_rho_NN*rho[i]
        #rearrangemet terms
        Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omega[i])**2
        Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho[i]  )**2
        Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sigma[i])**2
        #nucleons energy
        pre_p  = Inte(pre, (m_eff_p, Sigma_Np + Sigma_r_NN, mu_pr[i], T))
        pre_n  = Inte(pre, (m_eff_n, Sigma_Nn + Sigma_r_NN, mu_ne[i], T))
        #Total baryon energy
        pre_b = pre_p + pre_n
        pre_r = Sigma_r_NN * (n_pro[i] + n_neu[i])

        if not mat_sim:
            pre_b += Inte(pre, (m_l[0], 0, mu_el[i], T))

            if muon:
                pre_b += Inte(pre, (m_l[1], 0, mu_mu[i], T))

            if neutrin :
                pre_b += Inte(pre, (m_l[2], 0, mu_nu[i], T))

            if Lambda0 :
                m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0 - sig_s[i]*g_sgs_L0
                Sigma_L0 = g_omg_L0*omega[i] + g_phi_L0*phi[i]
                Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omega[i])**2
                Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sigma[i])**2
                Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sig_s[i])**2
                Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi[i]  )**2
                pre_b += Inte(pre, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0[i], T))
                pre_r += Sigma_r_L0 * n_lam[i]

            if SX_hyp :
                m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp
                Sigma_Sp = g_omg_Sp*omega[i] - g_rho_Sp*rho[i]
                Sigma_r_Sp  = (dg_omg_Sp/g_omg_Sp)*(m_m[1]*omega[i])**2
                Sigma_r_Sp += (dg_rho_Sp/g_rho_Sp)*(m_m[2]*rho[i]  )**2
                Sigma_r_Sp -= (dg_sig_Sp/g_sig_Sp)*(m_m[0]*sigma[i])**2
                pre_b += Inte(pre, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp[i], T))
                pre_r += Sigma_r_Sp * n_Sgp[i]
                #
                m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0
                Sigma_S0 = g_omg_S0*omega[i]
                Sigma_r_S0  = (dg_omg_S0/g_omg_S0)*(m_m[1]*omega[i])**2
                Sigma_r_S0 -= (dg_sig_S0/g_sig_S0)*(m_m[0]*sigma[i])**2
                pre_b += Inte(pre, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0[i], T))
                pre_r += Sigma_r_S0 * n_Sg0[i]
                #
                m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm
                Sigma_Sm = g_omg_Sm*omega[i] + g_rho_Sm*rho[i]
                Sigma_r_Sm  = (dg_omg_Sm/g_omg_Sm)*(m_m[1]*omega[i])**2
                Sigma_r_Sm += (dg_rho_Sm/g_rho_Sm)*(m_m[2]*rho[i]  )**2
                Sigma_r_Sm -= (dg_sig_Sm/g_sig_Sm)*(m_m[0]*sigma[i])**2
                pre_b += Inte(pre, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm[i], T))
                pre_r += Sigma_r_Sm * n_Sgm[i]
                #
                m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0
                Sigma_X0 = g_omg_X0*omega[i] - g_rho_X0*rho[i]
                Sigma_r_X0  = (dg_omg_X0/g_omg_X0)*(m_m[1]*omega[i])**2
                Sigma_r_X0 += (dg_rho_X0/g_rho_X0)*(m_m[2]*rho[i]  )**2
                Sigma_r_X0 -= (dg_sig_X0/g_sig_X0)*(m_m[0]*sigma[i])**2
                pre_b += Inte(pre, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0[i], T))
                pre_r += Sigma_r_X0 * n_Xi0[i]
                #
                m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm
                Sigma_Xm = g_omg_Xm*omega[i] + g_rho_Xm*rho[i]
                Sigma_r_Xm  = (dg_omg_Xm/g_omg_Xm)*(m_m[1]*omega[i])**2
                Sigma_r_Xm += (dg_rho_Xm/g_rho_Xm)*(m_m[2]*rho[i]  )**2
                Sigma_r_Xm -= (dg_sig_Xm/g_sig_Xm)*(m_m[0]*sigma[i])**2
                pre_b += Inte(pre, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm[i], T))
                pre_r += Sigma_r_Xm * n_Xim[i]


        #mesons pressure term
        pre_m2  = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        pre_m2 += -0.5*(m_m[3]**2*sig_s[i]**2 - m_m[4]**2*phi[i]**2)
        #system's total pressure
        press[i] = (1/3) * pre_b + pre_r + pre_m2

    return press


def Total_entropy(nb_dens, sigma, omega, rho, phi, sig_s,
                 mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                 mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T,
                 mat_sim, muon, Lambda0, SX_hyp, neutrin,
                 const_hyp, const_nuc):
    '''
    Computation of the system's total entropy

    Parameters
    ----------
    T : float
        temperature

    Retunrs
    -------
    entrop : 1darray
        array for the total entropy
    '''
    number_dens = len(nb_dens)
    #array for the entropy
    entrop = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):

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

        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sig_NN
        m_eff_n = m_n[1] - sigma[i]*g_sig_NN
        #energy terms
        Sigma_Np = g_omg_NN*omega[i] - g_rho_NN*rho[i]
        Sigma_Nn = g_omg_NN*omega[i] + g_rho_NN*rho[i]
        #rearrangemet terms
        Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omega[i])**2
        Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho[i]  )**2
        Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sigma[i])**2
        #nucleons energy
        I_p  = Inte(entropy, (m_eff_p, Sigma_Np + Sigma_r_NN, mu_pr[i], T))
        I_n  = Inte(entropy, (m_eff_n, Sigma_Nn + Sigma_r_NN, mu_ne[i], T))
        #Total baryon energy
        tot_ent = I_p + I_n

        if not mat_sim:
            tot_ent += Inte(entropy, (m_l[0], 0, mu_el[i], T))

            if muon:
                tot_ent += Inte(entropy, (m_l[1], 0, mu_mu[i], T))

            if neutrin :
                tot_ent += Inte(entropy, (m_l[2], 0, mu_nu[i], T))

            if Lambda0 :
                m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0 - sig_s[i]*g_sgs_L0
                Sigma_L0 = g_omg_L0*omega[i] + g_phi_L0*phi[i]
                Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omega[i])**2
                Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sigma[i])**2
                Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sig_s[i])**2
                Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi[i]  )**2
                tot_ent += Inte(entropy, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0[i], T))

            if SX_hyp :
                m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp
                Sigma_Sp = g_omg_Sp*omega[i] - g_rho_Sp*rho[i]
                Sigma_r_Sp  = (dg_omg_Sp/g_omg_Sp)*(m_m[1]*omega[i])**2
                Sigma_r_Sp += (dg_rho_Sp/g_rho_Sp)*(m_m[2]*rho[i]  )**2
                Sigma_r_Sp -= (dg_sig_Sp/g_sig_Sp)*(m_m[0]*sigma[i])**2
                tot_ent += Inte(entropy, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp[i], T))
                #
                m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0
                Sigma_S0 = g_omg_S0*omega[i]
                Sigma_r_S0  = (dg_omg_S0/g_omg_S0)*(m_m[1]*omega[i])**2
                Sigma_r_S0 -= (dg_sig_S0/g_sig_S0)*(m_m[0]*sigma[i])**2
                tot_ent += Inte(entropy, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0[i], T))
                #
                m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm
                Sigma_Sm = g_omg_Sm*omega[i] + g_rho_Sm*rho[i]
                Sigma_r_Sm  = (dg_omg_Sm/g_omg_Sm)*(m_m[1]*omega[i])**2
                Sigma_r_Sm += (dg_rho_Sm/g_rho_Sm)*(m_m[2]*rho[i]  )**2
                Sigma_r_Sm -= (dg_sig_Sm/g_sig_Sm)*(m_m[0]*sigma[i])**2
                tot_ent += Inte(entropy, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm[i], T))
                #
                m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0
                Sigma_X0 = g_omg_X0*omega[i] - g_rho_X0*rho[i]
                Sigma_r_X0  = (dg_omg_X0/g_omg_X0)*(m_m[1]*omega[i])**2
                Sigma_r_X0 += (dg_rho_X0/g_rho_X0)*(m_m[2]*rho[i]  )**2
                Sigma_r_X0 -= (dg_sig_X0/g_sig_X0)*(m_m[0]*sigma[i])**2
                tot_ent += Inte(entropy, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0[i], T))
                #
                m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm
                Sigma_Xm = g_omg_Xm*omega[i] + g_rho_Xm*rho[i]
                Sigma_r_Xm  = (dg_omg_Xm/g_omg_Xm)*(m_m[1]*omega[i])**2
                Sigma_r_Xm += (dg_rho_Xm/g_rho_Xm)*(m_m[2]*rho[i]  )**2
                Sigma_r_Xm -= (dg_sig_Xm/g_sig_Xm)*(m_m[0]*sigma[i])**2
                tot_ent += Inte(entropy, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm[i], T))

        entrop[i] = -tot_ent

    return entrop