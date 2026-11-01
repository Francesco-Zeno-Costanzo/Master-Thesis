"""
CAVEAT:
The code solves the TOV equations in the same way as the Lane-Emden equations.
So we suppose that the pressure is a polytrope of the density:

    P = K * rho**(1 + 1/n)

where K is the polytropic constant and n is the polytropic index (n=1.5).
So the equations will be solved in a dimensionless form:

    rho = rho_c * theta**n
    P   = K * rho_c**(1 + 1/n) * theta**(n + 1)

where rho_c is the central density and theta is the dimensionless density.
The TOV equations became:

                        (1 + (n + 1) * sigma * theta) * (mu + sigma * xi**3 * theta**(n + 1))
    dtheta/dxi =  -    ------------------------------------------------------------------------
                                  (xi**2 * (1 - 2 * (n + 1) * sigma * mu / xi))

    dmu/dxi    = xi**2 * theta**n * (1 + n * sigma * theta)

where xi is the dimensionless radius, mu is the dimensionless mass and sigma is:

    sigma = K * rho_c**(1/n) / c**2 = P_c/\rho_c * c**2

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as cst

#==================================================================
# Physical constants in SI units
#==================================================================
G    = cst.G                                               # Gravitational constant [m^3 kg^-1 s^-2]
c    = cst.c                                               # Speed of light [m/s]
hbar = cst.hbar                                            # Reduced Planck constant [J s]
m_n  = cst.m_n                                             # Neutron mass [kg]
MS   = 1.98847e30                                          # Solar mass [kg]
n    = 1.5                                                 # Polytropic index for non-relativistic degenerate neutron gas
K    = (3 * np.pi**2)**(2/3) * hbar**2 / (5 * m_n**(8/3))  # Polytropic constant [m^4 kg^-2/3 s^-2]

#==================================================================
# TOV equations in dimensionless form
#==================================================================

def tov(xi, theta, mu, sigma):
    """
    TOV equations in dimensionless form.

    Parameters
    ----------
    xi : float
        dimensionless radius.
    theta : float
        dimensionless density.
        rho = rho_c * theta**n
    mu : float
        dimensionless mass.
    sigma : float
        dimensionless parameter defined as:
        sigma = K * rho_c**(1/n) / c**2

        It measures the importance of relativistic effects associated with
        the central pressure relative to the rest-mass energy.

    Returns
    -------
    dtheta_dxi, dmu_dxi : float, float
        TOV equations derivatives
        
    Note
    ----
    The derivative of mu includes the contribution of the internal energy
    through the term (1 + n*sigma*theta).
    """

    # Avoid fractional powers of negative theta at the surface.
    th = max(theta, 0.0)

    
    rel_fact = (
        1 - 2 * (n + 1) * sigma * mu / xi
    )

    if rel_fact <= 0:
        raise RuntimeError(
            "Integration has reached a non-physical region."
        )

    dtheta_dxi = -(
          (1  + (n + 1) * sigma * th)
        * (mu + sigma   * xi**3 * th**(n + 1))
        / (xi**2 * rel_fact)
    )

    dmu_dxi = xi**2 * th**n * (1 + n * sigma * th)

    return dtheta_dxi, dmu_dxi

#==================================================================
# Runge-Kutta 4th order integration of the TOV equations
#==================================================================

def RK4(rho_c, dxi=1e-3, xi_max=100.0):
    '''
    Integrate the TOV equations using the 4th order Runge-Kutta method.

    Parameters
    ----------
    rho_c : float
        Central mass density [kg/m^3].
    dxi : float, optional
        Step size for the integration (default is 1e-3).
    xi_max : float, optional
        Maximum value of the dimensionless radius (default is 100.0).

    Returns
    -------
    dict :
        xi, theta, mu: arrays of dimensionless radius, density, and mass.
        radius       : array of physical radius in meters.
        mass         : array of physical mass in kilograms.
        density      : array of physical density in kg/m^3.
        pressure     : array of physical pressure in Pascals.
        R_m          : physical radius of the star in meters.
        M_kg         : physical mass of the star in kilograms.
    '''

    sigma = K * rho_c**(1/n) / c**2

    # Physical length and mass scales
    # r = len_scale * xi
    len_scale  = np.sqrt((n + 1) * K * rho_c**(1/n - 1) / (4 * np.pi * G))

    # m = mass_scale * mu
    mass_scale = 4 * np.pi * len_scale**3 * rho_c

    #=== Initial conditions ===#
    # Start slightly away from the singular centre xi = 0.
    # Other condition came from analytical expansion
    # of the TOV equations around xi = 0.
    xi    = 1e-8  
    coef  = (1 + (n + 1) * sigma) * ((1 + n * sigma) / 3 + sigma)
    theta = 1 - 0.5 * coef * xi**2
    mu    = (1 + n * sigma) * xi**3 / 3


    X = [xi]
    T = [theta]
    M = [mu]

    while xi < xi_max:

        k1t, k1m = tov(xi,         theta,             mu,             sigma)
        k2t, k2m = tov(xi + dxi/2, theta + k1t*dxi/2, mu + k1m*dxi/2, sigma)
        k3t, k3m = tov(xi + dxi/2, theta + k2t*dxi/2, mu + k2m*dxi/2, sigma)
        k4t, k4m = tov(xi + dxi,   theta + k3t*dxi,   mu + k3m*dxi,   sigma)

        theta_new = theta + (dxi/6) * (k1t + 2*k2t + 2*k3t + k4t)
        mu_new    = mu    + (dxi/6) * (k1m + 2*k2m + 2*k3m + k4m)
        xi_new    = xi    + dxi

        # Search for the surface of the star where theta crosses zero.
        if theta_new <= 0:
            # Linear interpolation to find the surface more accurately.
            frac    = theta / (theta - theta_new)
            xi_surf = xi + frac * dxi
            mu_surf = mu + frac * (mu_new - mu)

            X.append(xi_surf)
            T.append(0.0)
            M.append(mu_surf)
            break
        
        # Update variables for the next iteration
        xi, theta, mu = xi_new, theta_new, mu_new
        X.append(xi)
        T.append(theta)
        M.append(mu)

    # Convert dimensionless quantities to physical units
    X = np.array(X)
    T = np.array(T)
    M = np.array(M)

    radius   = len_scale * X
    mass     = mass_scale * M
    density  = rho_c * T**n
    pressure = K * rho_c**(1 + 1/n) * T**(n + 1)

    return {
        "xi":       X,
        "theta":    T,
        "mu":       M,
        "radius":   radius,
        "mass":     mass,
        "density":  density,
        "pressure": pressure,
        "R_m":      radius[-1],
        "M_kg":     mass[-1]
    }


#==================================================================
# One stellar solution
#==================================================================

rho_c = 3e17  # Central density [kg/m^3]
star  = RK4(rho_c)

print(f"Central density: {rho_c:.3e} kg/m^3")
print(f"Radius: {star['R_m']/1000:.3f} km")
print(f"Mass: {star['M_kg']/MS:.3f} solar masses")


plt.figure(1)
plt.suptitle('Mass and pressure as a function of radius')
plt.subplot(121)
plt.grid()
plt.xlabel('r [km]', fontsize=15)
plt.ylabel(r"$M/M_\odot$", fontsize=15)
plt.plot(star['radius']/1000, star['mass']/MS)
plt.subplot(122)
plt.grid()
plt.xlabel('r [km]', fontsize=15)
plt.ylabel('P', fontsize=15)
plt.plot(star['radius']/1000, star['pressure'])
plt.xlabel('r [km]', fontsize=15)
plt.ylabel('P', fontsize=15)
plt.show()

#========================================================
# Mass - Radius curve
#========================================================
rho_values = np.logspace(14, 20, 100)

radii  = []
masses = []

for rho_c in rho_values:
    star = RK4(rho_c)

    radii.append(star["R_m"] / 1000)
    masses.append(star["M_kg"] / MS)

plt.plot(radii, masses)
plt.xlabel("R [km]", fontsize=15)
plt.ylabel(r"$M/M_\odot$", fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
