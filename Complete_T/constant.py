"""
All important constant
"""

import numpy as np

#==================================================================
# Physical constant and coupling constant
#==================================================================

#\hbar/c in MeV fermi
hbc = 197.327

#saturation numerical densitity
nb_sat = 0.152

#baryon masses
mass_p  = 938.27203
mass_n  = 939.56536
mass_L  = 1115.683
mass_Sp = 1189.37
mass_S0 = 1192.642
mass_Sm = 1197.45
mass_X0 = 1314.86
mass_Xm = 1321.71
m_n = np.array([mass_p,  mass_n,  mass_L,  mass_Sp,
                mass_S0, mass_Sm, mass_X0, mass_Xm])/hbc

#meson masses
m_sig = 550.1238
m_omg = 783.0000
m_rho = 763.0000
m_sgs = 980.0
m_phi = 1020
m_m   = np.array([m_sig, m_omg, m_rho, m_sgs, m_phi])/hbc

#lepton masses
m_el = 0.51099907
m_mu = 105.6583755
m_nu = 0
m_l  = np.array([m_el, m_mu, m_nu])/hbc

#coupling constants at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #we already considered 1/2
g_sat_NN = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#hyperonic couplig costant Lambda-meson
L_sig_DD = 0.65869756
L_sig_ND = 0.61498048
L_sgs_DD = 0.535
L_sgs_ND = 0.538
x_sig_L0 = L_sig_ND
x_omg_L0 = 2/3
x_phi_L0 = - np.sqrt(2)/3
x_sgs_L0 = L_sgs_ND

g_sig_L_sat = x_sig_L0 * g_sg_sat
g_omg_L_sat = x_omg_L0 * g_om_sat
g_phi_L_sat = x_phi_L0 * g_om_sat
g_sgs_L_sat = x_sgs_L0 * g_sg_sat
#in order            sigma,       omega,       phi           sigma*
g_sat_L0 = np.array([g_sig_L_sat, g_omg_L_sat, g_phi_L_sat, g_sgs_L_sat])

#hyperonic couplig costant Sigma_p-meson
S_sig_30_DD = 0.50447765
S_sig_30_ND = 0.46076056
S_sig_00_DD = 0.57910019
S_sig_00_ND = 0.5353831
x_sig_S = S_sig_00_ND
x_omg_S = 2/3
x_rho_S = 2

g_sig_Sp_sat = x_sig_S * g_sg_sat
g_omg_Sp_sat = x_omg_S * g_om_sat
g_rho_Sp_sat = x_rho_S * g_rh_sat
g_sat_Sp = np.array([g_sig_Sp_sat, g_omg_Sp_sat, g_rho_Sp_sat])

#hyperonic couplig costant Sigma_0-meson
g_sig_S0_sat = x_sig_S * g_sg_sat
g_omg_S0_sat = x_omg_S * g_om_sat
g_sat_S0 = np.array([g_sig_S0_sat, g_omg_S0_sat])

#hyperonic couplig costant Sigma_m-meson
g_sig_Sm_sat = x_sig_S * g_sg_sat
g_omg_Sm_sat = x_omg_S * g_om_sat
g_rho_Sm_sat = x_rho_S * g_rh_sat
g_sat_Sm = np.array([g_sig_Sm_sat, g_omg_Sm_sat, g_rho_Sm_sat])

#hyperonic couplig costant Xi_0-meson
X_sig_DD = 0.37988278
X_sig_ND = 0.30251540
x_sig_X = X_sig_ND
x_omg_X = 1/3
x_rho_X = 1

g_sig_X0_sat = x_sig_X * g_sg_sat
g_omg_X0_sat = x_omg_X * g_om_sat
g_rho_X0_sat = x_rho_X * g_rh_sat
g_sat_X0 = np.array([g_sig_X0_sat, g_omg_X0_sat, g_rho_X0_sat])

#hyperonic couplig costant Xi_m-meson
g_sig_Xm_sat = x_sig_X * g_sg_sat
g_omg_Xm_sat = x_omg_X * g_om_sat
g_rho_Xm_sat = x_rho_X * g_rh_sat
g_sat_Xm = np.array([g_sig_Xm_sat, g_omg_Xm_sat, g_rho_Xm_sat])


#array to model the dependence on the coupling constants
#in order     sigma,  omega,  rho
a = np.array([1.3881, 1.3892, 0.5647])
b = np.array([1.0943, 0.9240])
c = np.array([1.7057, 1.4620])
d = np.array([0.4421, 0.4775])