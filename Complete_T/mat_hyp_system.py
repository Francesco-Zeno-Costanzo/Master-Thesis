"""
Code with three type of system for beta equilibrium at finite T:

1) system_lam--> neutrons, protons, electrons, muon and lambda
2) system_hyp--> system_lam + all Sigma and all Xi
3) system_all--> system_hyp + neutrinos

"""
import numpy as np

from constant import *
from utilities import *


#==============================================================
# neutrons, protons, electrons, muon and lambda
#==============================================================


def system_lam(V, nb,
               g_sig_NN,  g_omg_NN,  g_rho_NN,
               dg_sig_NN, dg_omg_NN, dg_rho_NN,
               T,
               g_sig_L0,  g_omg_L0,  g_phi_L0,  g_sgs_L0,
               dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0):
    """
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density
    T : float
        temperature
    g_meson_baryon : float
        coupling constant
    dg_meson_baryon : float
        derivative of coupling constant

    Returns
    -------
    list of equations to solve
    ri = 0
    """
    sig, omg, rho                = V[0:3]
    p, n, e, mu_p, mu_n, mu_e    = V[3:9]
    m, mu_m, sgs, phi, L0, mu_L0 = V[9:15]


    #effective masses
    m_eff_p  = m_n[0] - sig*g_sig_NN
    m_eff_n  = m_n[1] - sig*g_sig_NN
    m_eff_L0 = m_n[2] - sig*g_sig_L0 - sgs*g_sgs_L0
    #Energy terms
    Sigma_Np = g_omg_NN*omg - g_rho_NN*rho
    Sigma_Nn = g_omg_NN*omg + g_rho_NN*rho
    Sigma_L0 = g_omg_L0*omg + g_phi_L0*phi
    #Reanrangement terms
    Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omg)**2
    Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho)**2
    Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sig)**2
    Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omg)**2
    Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sig)**2
    Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sgs)**2
    Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi)**2
    #numerical integrals for sigma
    I_p  = Inte(ns, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T))
    I_n  = Inte(ns, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T))
    I_L0 = Inte(ns, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T))
    #total integrals for sigma
    I_sg_n  = g_sig_NN*I_p + g_sig_NN*I_n + g_sig_L0*I_L0
    #density for omega
    ND_omg  = g_omg_NN*n + g_omg_NN*p + g_omg_L0*L0
    #density for rho
    ND_rho = g_rho_NN*(n - p)
    #total integrals for sigma*
    I_sg_s = g_sgs_L0*I_L0
    #density for phi
    ND_phi = g_phi_L0*L0
    #fermi dirac integrals baryons
    I_f_p_p  = Inte(fermi_dirac_p, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_p_a  = Inte(fermi_dirac_a, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_n_p  = Inte(fermi_dirac_p, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_n_a  = Inte(fermi_dirac_a, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_L0_p = Inte(fermi_dirac_p, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_L0_a = Inte(fermi_dirac_a, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    #fermi dirac integrals for leptons
    I_f_e_p = Inte(fermi_dirac_p, (m_l[0], 0, mu_e, T, True))
    I_f_e_a = Inte(fermi_dirac_a, (m_l[0], 0, mu_e, T, True))
    I_f_m_p = Inte(fermi_dirac_p, (m_l[1], 0, mu_m, T, True))
    I_f_m_a = Inte(fermi_dirac_a, (m_l[1], 0, mu_m, T, True))
    #equation to solve
    r1 = sig - 1/m_m[0]**2 * I_sg_n
    r2 = omg - 1/m_m[1]**2 * ND_omg
    r3 = rho - 1/m_m[2]**2 * ND_rho
    r4 = sgs - 1/m_m[3]**2 * I_sg_s
    r5 = phi - 1/m_m[4]**2 * ND_phi
    r6 = nb - (n + p + L0)
    r7 = p + - (e + m)
    r8  = mu_n - mu_p - mu_e
    r9  = mu_e - mu_m
    r10 = mu_n - mu_L0
    r11 = p - (I_f_p_p - I_f_p_a)
    r12 = n - (I_f_n_p - I_f_n_a)
    r13 = e - (I_f_e_p - I_f_e_a)
    r14 = m - (I_f_m_p - I_f_m_a)
    r15 = L0 - (I_f_L0_p - I_f_L0_a)


    R = [r1,  r2,  r3,  r4,  r5, r6,  r7,  r8,  r9,  r10, r11, r12, r13, r14, r15]

    return R


#==============================================================
# All octet + elecntrons and muons
#==============================================================


def system_hyp(V, nb,
               g_sig_NN,  g_omg_NN,  g_rho_NN,
               dg_sig_NN, dg_omg_NN, dg_rho_NN,
               T,
               g_sig_L0,  g_omg_L0,  g_phi_L0,  g_sgs_L0,
               dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
               g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
               dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
               g_sig_S0,  g_omg_S0,
               dg_sig_S0, dg_omg_S0,
               g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
               dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
               g_sig_X0,  g_omg_X0,  g_rho_X0,
               dg_sig_X0, dg_omg_X0, dg_rho_X0,
               g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
               dg_sig_Xm, dg_omg_Xm, dg_rho_Xm):
    '''
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density
    T : float
        temperature
    g_meson_baryon : float
        coupling constant
    dg_meson_baryon : float
        derivative of coupling constant

    Returns
    -------
    list of equations to solve
    ri = 0
    '''
    sig, omg, rho                   = V[0:3]
    p, n, e, mu_p, mu_n, mu_e       = V[3:9]
    m, mu_m, sgs, phi, L0, mu_L0    = V[9:15]
    Sp, S0, Sm, mu_Sp, mu_S0, mu_Sm = V[15:21]
    X0, Xm, mu_X0, mu_Xm            = V[21:25]

    #effective masses
    m_eff_p  = m_n[0] - sig*g_sig_NN
    m_eff_n  = m_n[1] - sig*g_sig_NN
    m_eff_L0 = m_n[2] - sig*g_sig_L0 - sgs*g_sgs_L0
    m_eff_Sp = m_n[3] - sig*g_sig_Sp
    m_eff_S0 = m_n[4] - sig*g_sig_S0
    m_eff_Sm = m_n[5] - sig*g_sig_Sm
    m_eff_X0 = m_n[6] - sig*g_sig_X0
    m_eff_Xm = m_n[7] - sig*g_sig_Xm
    #energy terms
    Sigma_Np = g_omg_NN*omg - g_rho_NN*rho
    Sigma_Nn = g_omg_NN*omg + g_rho_NN*rho
    Sigma_L0 = g_omg_L0*omg + g_phi_L0*phi
    Sigma_Sp = g_omg_Sp*omg - g_rho_Sp*rho
    Sigma_S0 = g_omg_S0*omg
    Sigma_Sm = g_omg_Sm*omg + g_rho_Sm*rho
    Sigma_X0 = g_omg_X0*omg - g_rho_X0*rho
    Sigma_Xm = g_omg_Xm*omg + g_rho_Xm*rho
    #rearrangemet terms
    Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omg)**2
    Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho)**2
    Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sig)**2
    #
    Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omg)**2
    Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sig)**2
    Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sgs)**2
    Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi)**2
    #
    Sigma_r_Sp  = (dg_omg_Sp/g_omg_Sp)*(m_m[1]*omg)**2
    Sigma_r_Sp += (dg_rho_Sp/g_rho_Sp)*(m_m[2]*rho)**2
    Sigma_r_Sp -= (dg_sig_Sp/g_sig_Sp)*(m_m[0]*sig)**2
    #
    Sigma_r_S0  = (dg_omg_S0/g_omg_S0)*(m_m[1]*omg)**2
    Sigma_r_S0 -= (dg_sig_S0/g_sig_S0)*(m_m[0]*sig)**2
    #
    Sigma_r_Sm  = (dg_omg_Sm/g_omg_Sm)*(m_m[1]*omg)**2
    Sigma_r_Sm += (dg_rho_Sm/g_rho_Sm)*(m_m[2]*rho)**2
    Sigma_r_Sm -= (dg_sig_Sm/g_sig_Sm)*(m_m[0]*sig)**2
    #
    Sigma_r_X0  = (dg_omg_X0/g_omg_X0)*(m_m[1]*omg)**2
    Sigma_r_X0 += (dg_rho_X0/g_rho_X0)*(m_m[2]*rho)**2
    Sigma_r_X0 -= (dg_sig_X0/g_sig_X0)*(m_m[0]*sig)**2
    #
    Sigma_r_Xm  = (dg_omg_Xm/g_omg_Xm)*(m_m[1]*omg)**2
    Sigma_r_Xm += (dg_rho_Xm/g_rho_Xm)*(m_m[2]*rho)**2
    Sigma_r_Xm -= (dg_sig_Xm/g_sig_Xm)*(m_m[0]*sig)**2
    #numerical integrals for sigma
    I_p  = Inte(ns, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T))
    I_n  = Inte(ns, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T))
    I_L0 = Inte(ns, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T))
    I_Sp = Inte(ns, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T))
    I_S0 = Inte(ns, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T))
    I_Sm = Inte(ns, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T))
    I_X0 = Inte(ns, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T))
    I_Xm = Inte(ns, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T))
    #total integrals for sigma
    I_sg_n  = g_sig_NN*I_p  + g_sig_NN*I_n  + g_sig_L0*I_L0 + g_sig_Sp*I_Sp
    I_sg_n += g_sig_S0*I_S0 + g_sig_Sm*I_Sm + g_sig_X0*I_X0 + g_sig_Xm*I_Xm
    #density for omega
    ND_omg  = g_omg_NN*n  + g_omg_NN*p  + g_omg_L0*L0 + g_omg_Sp*Sp
    ND_omg += g_omg_S0*S0 + g_omg_Sm*Sm + g_omg_X0*X0 + g_omg_Xm*Xm
    #density for rho
    ND_rho = g_rho_NN*(n - p) - g_rho_Sp*Sp + g_rho_Sm*Sm - g_rho_X0*X0 + g_rho_Xm*Xm
    #total integrals for sigma*
    I_sg_s = g_sgs_L0*I_L0
    #density for phi
    ND_phi = g_phi_L0*L0
    #fermi dirac integrals baryons
    I_f_p_p  = Inte(fermi_dirac_p, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_p_a  = Inte(fermi_dirac_a, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_n_p  = Inte(fermi_dirac_p, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_n_a  = Inte(fermi_dirac_a, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_L0_p = Inte(fermi_dirac_p, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_L0_a = Inte(fermi_dirac_a, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_Sp_p = Inte(fermi_dirac_p, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T, True))
    I_f_Sp_a = Inte(fermi_dirac_a, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T, True))
    I_f_S0_p = Inte(fermi_dirac_p, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T, True))
    I_f_S0_a = Inte(fermi_dirac_a, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T, True))
    I_f_Sm_p = Inte(fermi_dirac_p, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T, True))
    I_f_Sm_a = Inte(fermi_dirac_a, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T, True))
    I_f_X0_p = Inte(fermi_dirac_p, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T, True))
    I_f_X0_a = Inte(fermi_dirac_a, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T, True))
    I_f_Xm_p = Inte(fermi_dirac_p, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T, True))
    I_f_Xm_a = Inte(fermi_dirac_a, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T, True))
    #fermi dirac integrals for leptons
    I_f_e_p = Inte(fermi_dirac_p, (m_l[0], 0, mu_e, T, True))
    I_f_e_a = Inte(fermi_dirac_a, (m_l[0], 0, mu_e, T, True))
    I_f_m_p = Inte(fermi_dirac_p, (m_l[1], 0, mu_m, T, True))
    I_f_m_a = Inte(fermi_dirac_a, (m_l[1], 0, mu_m, T, True))
    #equation to solve
    r1 = sig - 1/m_m[0]**2 * I_sg_n
    r2 = omg - 1/m_m[1]**2 * ND_omg
    r3 = rho - 1/m_m[2]**2 * ND_rho
    r4 = sgs - 1/m_m[3]**2 * I_sg_s
    r5 = phi - 1/m_m[4]**2 * ND_phi
    r6 = nb - (n + p + L0 + Sp + S0 + Sm + X0 + Xm)
    r7 = (p + Sp) - (e + m + Sm + Xm)
    r8  = mu_n - mu_p - mu_e
    r9  = mu_e - mu_m
    r10 = mu_n - mu_L0
    r11 = mu_p - mu_Sp
    r12 = mu_n - mu_S0
    r13 = mu_n + mu_e - mu_Sm
    r14 = mu_n - mu_X0
    r15 = mu_n + mu_e - mu_Xm
    r16 = p - (I_f_p_p - I_f_p_a)
    r17 = n - (I_f_n_p - I_f_n_a)
    r18 = e - (I_f_e_p - I_f_e_a)
    r19 = m - (I_f_m_p - I_f_m_a)
    r20 = L0 - (I_f_L0_p - I_f_L0_a)
    r21 = Sp - (I_f_Sp_p - I_f_Sp_a)
    r22 = S0 - (I_f_S0_p - I_f_S0_a)
    r23 = Sm - (I_f_Sm_p - I_f_Sm_a)
    r24 = X0 - (I_f_X0_p - I_f_X0_a)
    r25 = Xm - (I_f_Xm_p - I_f_Xm_a)

    R = [r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  r9,  r10,
         r11, r12, r13, r14, r15, r16, r17, r18, r19, r20,
         r21, r22, r23, r24, r25
        ]

    return R


#==============================================================
# All octet + elecntrons, muons and neutrinos
#==============================================================


def system_all(V, nb,
               g_sig_NN,  g_omg_NN,  g_rho_NN,
               dg_sig_NN, dg_omg_NN, dg_rho_NN,
               T,
               g_sig_L0,  g_omg_L0,  g_phi_L0,  g_sgs_L0,
               dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
               g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
               dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
               g_sig_S0,  g_omg_S0,
               dg_sig_S0, dg_omg_S0,
               g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
               dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
               g_sig_X0,  g_omg_X0,  g_rho_X0,
               dg_sig_X0, dg_omg_X0, dg_rho_X0,
               g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
               dg_sig_Xm, dg_omg_Xm, dg_rho_Xm):
    '''
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density
    T : float
        temperature
    g_meson_baryon : float
        coupling constant
    dg_meson_baryon : float
        derivative of coupling constant

    Returns
    -------
    list of equations to solve
    ri = 0
    '''
    sig, omg, rho                      = V[0:3]
    p, n, e, mu_p, mu_n, mu_e, m, mu_m = V[3:11]
    nu, mu_nu, sgs, phi, L0, mu_L0     = V[11:17]
    Sp, S0, Sm, mu_Sp, mu_S0, mu_Sm    = V[17:23]
    X0, Xm, mu_X0, mu_Xm               = V[23:27]

    #effective masses
    m_eff_p  = m_n[0] - sig*g_sig_NN
    m_eff_n  = m_n[1] - sig*g_sig_NN
    m_eff_L0 = m_n[2] - sig*g_sig_L0 - sgs*g_sgs_L0
    m_eff_Sp = m_n[3] - sig*g_sig_Sp
    m_eff_S0 = m_n[4] - sig*g_sig_S0
    m_eff_Sm = m_n[5] - sig*g_sig_Sm
    m_eff_X0 = m_n[6] - sig*g_sig_X0
    m_eff_Xm = m_n[7] - sig*g_sig_Xm
    #energy terms
    Sigma_Np = g_omg_NN*omg - g_rho_NN*rho
    Sigma_Nn = g_omg_NN*omg + g_rho_NN*rho
    Sigma_L0 = g_omg_L0*omg + g_phi_L0*phi
    Sigma_Sp = g_omg_Sp*omg - g_rho_Sp*rho
    Sigma_S0 = g_omg_S0*omg
    Sigma_Sm = g_omg_Sm*omg + g_rho_Sm*rho
    Sigma_X0 = g_omg_X0*omg - g_rho_X0*rho
    Sigma_Xm = g_omg_Xm*omg + g_rho_Xm*rho
    #rearrangemet terms
    Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omg)**2
    Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho)**2
    Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sig)**2
    #
    Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omg)**2
    Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sig)**2
    Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sgs)**2
    Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi)**2
    #
    Sigma_r_Sp  = (dg_omg_Sp/g_omg_Sp)*(m_m[1]*omg)**2
    Sigma_r_Sp += (dg_rho_Sp/g_rho_Sp)*(m_m[2]*rho)**2
    Sigma_r_Sp -= (dg_sig_Sp/g_sig_Sp)*(m_m[0]*sig)**2
    #
    Sigma_r_S0  = (dg_omg_S0/g_omg_S0)*(m_m[1]*omg)**2
    Sigma_r_S0 -= (dg_sig_S0/g_sig_S0)*(m_m[0]*sig)**2
    #
    Sigma_r_Sm  = (dg_omg_Sm/g_omg_Sm)*(m_m[1]*omg)**2
    Sigma_r_Sm += (dg_rho_Sm/g_rho_Sm)*(m_m[2]*rho)**2
    Sigma_r_Sm -= (dg_sig_Sm/g_sig_Sm)*(m_m[0]*sig)**2
    #
    Sigma_r_X0  = (dg_omg_X0/g_omg_X0)*(m_m[1]*omg)**2
    Sigma_r_X0 += (dg_rho_X0/g_rho_X0)*(m_m[2]*rho)**2
    Sigma_r_X0 -= (dg_sig_X0/g_sig_X0)*(m_m[0]*sig)**2
    #
    Sigma_r_Xm  = (dg_omg_Xm/g_omg_Xm)*(m_m[1]*omg)**2
    Sigma_r_Xm += (dg_rho_Xm/g_rho_Xm)*(m_m[2]*rho)**2
    Sigma_r_Xm -= (dg_sig_Xm/g_sig_Xm)*(m_m[0]*sig)**2
    #numerical integrals for sigma
    I_p  = Inte(ns, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T))
    I_n  = Inte(ns, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T))
    I_L0 = Inte(ns, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T))
    I_Sp = Inte(ns, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T))
    I_S0 = Inte(ns, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T))
    I_Sm = Inte(ns, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T))
    I_X0 = Inte(ns, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T))
    I_Xm = Inte(ns, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T))
    #total integrals for sigma
    I_sg_n  = g_sig_NN*I_p  + g_sig_NN*I_n  + g_sig_L0*I_L0 + g_sig_Sp*I_Sp
    I_sg_n += g_sig_S0*I_S0 + g_sig_Sm*I_Sm + g_sig_X0*I_X0 + g_sig_Xm*I_Xm
    #density for omega
    ND_omg  = g_omg_NN*n  + g_omg_NN*p  + g_omg_L0*L0 + g_omg_Sp*Sp
    ND_omg += g_omg_S0*S0 + g_omg_Sm*Sm + g_omg_X0*X0 + g_omg_Xm*Xm
    #density for rho
    ND_rho = g_rho_NN*(n - p) - g_rho_Sp*Sp + g_rho_Sm*Sm - g_rho_X0*X0 + g_rho_Xm*Xm
    #total integrals for sigma*
    I_sg_s = g_sgs_L0*I_L0
    #density for phi
    ND_phi = g_phi_L0*L0
    #fermi dirac integrals baryons
    I_f_p_p  = Inte(fermi_dirac_p, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_p_a  = Inte(fermi_dirac_a, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_n_p  = Inte(fermi_dirac_p, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_n_a  = Inte(fermi_dirac_a, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_L0_p = Inte(fermi_dirac_p, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_L0_a = Inte(fermi_dirac_a, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_Sp_p = Inte(fermi_dirac_p, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T, True))
    I_f_Sp_a = Inte(fermi_dirac_a, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T, True))
    I_f_S0_p = Inte(fermi_dirac_p, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T, True))
    I_f_S0_a = Inte(fermi_dirac_a, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T, True))
    I_f_Sm_p = Inte(fermi_dirac_p, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T, True))
    I_f_Sm_a = Inte(fermi_dirac_a, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T, True))
    I_f_X0_p = Inte(fermi_dirac_p, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T, True))
    I_f_X0_a = Inte(fermi_dirac_a, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T, True))
    I_f_Xm_p = Inte(fermi_dirac_p, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T, True))
    I_f_Xm_a = Inte(fermi_dirac_a, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T, True))
    #fermi dirac integrals for leptons
    I_f_e_p  = Inte(fermi_dirac_p, (m_l[0], 0, mu_e, T, True))
    I_f_e_a  = Inte(fermi_dirac_a, (m_l[0], 0, mu_e, T, True))
    I_f_m_p  = Inte(fermi_dirac_p, (m_l[1], 0, mu_m, T, True))
    I_f_m_a  = Inte(fermi_dirac_a, (m_l[1], 0, mu_m, T, True))
    I_f_nu_p = Inte(fermi_dirac_p, (m_l[2], 0, mu_nu, T, True))
    I_f_nu_a = Inte(fermi_dirac_a, (m_l[2], 0, mu_nu, T, True))
    #equation to solve
    r1 = sig - 1/m_m[0]**2 * I_sg_n
    r2 = omg - 1/m_m[1]**2 * ND_omg
    r3 = rho - 1/m_m[2]**2 * ND_rho
    r4 = sgs - 1/m_m[3]**2 * I_sg_s
    r5 = phi - 1/m_m[4]**2 * ND_phi
    r6 = nb - (n + p + L0 + Sp + S0 + Sm + X0 + Xm)
    r7 = (p + Sp) - (e + m + Sm + Xm)
    r8 = nu + e - 0.33*nb
    r9  = mu_n - mu_p - mu_e + mu_nu
    r10  = mu_e - mu_m
    r11 = mu_n - mu_L0
    r12 = mu_p - mu_Sp
    r13 = mu_n - mu_S0
    r14 = mu_n + mu_e - mu_Sm
    r15 = mu_n - mu_X0
    r16 = mu_n + mu_e - mu_Xm
    r17 = p - (I_f_p_p - I_f_p_a)
    r18 = n - (I_f_n_p - I_f_n_a)
    r19 = e - (I_f_e_p - I_f_e_a)
    r20 = m - (I_f_m_p - I_f_m_a)
    r21 = L0 - (I_f_L0_p - I_f_L0_a)
    r22 = Sp - (I_f_Sp_p - I_f_Sp_a)
    r23 = S0 - (I_f_S0_p - I_f_S0_a)
    r24 = Sm - (I_f_Sm_p - I_f_Sm_a)
    r25 = X0 - (I_f_X0_p - I_f_X0_a)
    r26 = Xm - (I_f_Xm_p - I_f_Xm_a)
    r27 = nu - (I_f_nu_p - I_f_nu_a)

    R = [r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  r9,  r10,
         r11, r12, r13, r14, r15, r16, r17, r18, r19, r20,
         r21, r22, r23, r24, r25, r26, r27
        ]

    return R

