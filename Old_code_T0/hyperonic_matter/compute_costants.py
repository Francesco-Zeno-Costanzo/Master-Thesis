import numpy as np
import matplotlib.pyplot as plt

#\hbar/c in MeV fermi
hbc = 197.327

#saturation numerical densitity
nb_sat = 0.152

#hyperon single particle energy
U_L = -32
U_S = 0#30
U_X = -14
U_H = np.array([U_L, U_S, U_X])/hbc

#coupling constants at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #we already considered 1/2
g_sat = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#array to model the dependence on the coupling constants
#in order     sigma,  omega,  rho
a = np.array([1.3881, 1.3892, 0.5647])
b = np.array([1.0943, 0.9240])
c = np.array([1.7057, 1.4620])
d = np.array([0.4421, 0.4775])

#meson masses
m_sg  = 550.1238
m_om  = 783.0000
m_rh  = 763.0000
m_sgs = 980.0
m_phi = 1020
m_m = np.array([m_sg, m_om, m_rh, m_sgs, m_phi])/hbc


sigma_n0 = 0.19330389314799515
omega_n0 = 0.12567426596661097

#==============================================================
# Coupling with Lambda, Sigma and Xi without density dependence
#==============================================================

g_omg_H = np.array([2/3, 2/3, 1/3]) * g_om_sat          #omega hyperon
g_rho_H = np.array([0,   2,   1  ]) * g_rh_sat          #rho   hyperon
g_sig_H = (-U_H + g_omg_H*omega_n0)/sigma_n0            #sigma lambda
g_phi_H = np.array([- np.sqrt(2)/3, 0, 0]) * g_om_sat   #phi   lambda
g_sgs_L = - np.sqrt(2)/3 * g_sg_sat
print(g_sig_H)
print(g_sig_H/g_sg_sat)

#===========================================================
# Coupling with Lambda, Sigma and Xi with density dependence
#===========================================================

def f(n, i, derive=False):
    """
    function that models the behaviour of the
    coupling constants depending on nb
    for the sigma meson and the omega meson
    It returns the derivative depending on the value of derive

    Parameters
    ----------
    n : float or 1darray
        total baryon density
    i : int
        it is a flag
        i = 0 sigma
        i = 1 omega
    derive : boolen
        it is a flag if True returns the derivative

    Return
    ----------
    derive=False
        g_n : float or 1darray
            g(nb/nb_sat)
    derive=True
        dg_n :float or 1d array
            dg/dnb (nb/nb_sat)
    """
    nb_sat = 0.152
    x = n/nb_sat

    if not derive :
        num = 1 + b[i]*(x + d[i])**2
        den = 1 + c[i]*(x + d[i])**2
        g_n = a[i] * num/den
        return g_n

    else :
        num = 2*a[i]*(b[i] - c[i])*nb_sat**2 * (n + d[i]*nb_sat)
        den = (nb_sat**2 + c[i]*(n + d[i]*nb_sat)**2)**2
        dg_n = num/den
        return dg_n



dg_omg_L = g_omg_H[0]*f(nb_sat, 1, derive=True)

g_sig_L = (-(U_H[0] - (g_omg_H[0]+dg_omg_L*nb_sat)*omega_n0) - f(nb_sat, 0, derive=True)*(m_m[0]*sigma_n0)**2)/sigma_n0
print(g_sig_L/g_sg_sat)

