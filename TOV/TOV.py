"""
Code for the solution of the Tov equations; the input
is an Equation of State in the following units:

1) energy density [MeV/fm^3]
2) pressure [MeV/fm^3]
3) baryonic density [1/fm^3]

The output can be the structure of a single star or of a
sequence of stars to compute the mass-radius relation;
Futhermore the following quantities are computed for every star:

1) gravitational mass [M_sun]
2) proper        mass [M_sun]
3) baryonic      mass [M_sun]
4) Gravitational binding energy [x 10^53 ergs]
5) Internal binding energy      [x 10^53 ergs]
6) Total binding energy         [x 10^53 ergs]
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
from interp import CubicSpline

ene_constant1       = 2.885838793e-6      # conversion factors to make ene a dimensionless variable
pre_constant2       = 2.885843707e-6      # conversion factors to make pre a dimensionless variable
m_pro               = 938.27231           # proton  mass
m_neu               = 939.56563           # neutron mass
uma                 = (m_pro + m_neu)/2   # unified atomic mass unit
MeV_to_g            = 1.7827e-27          # from MeV to grams
M_sun               = 1.9889225e33        # solar mass in grams
r_sun               = 1.476625529         # 0.5 * Schwarzschild radius of the Sun in Km (mass of the sun in km)
c                   = 2.9979245e10        # speed of light in cm/s
pn_to_1036_dyne_cm2 = 5.551975999e2       # from pure number to 10^36 dyne/cm2
typical_energy      = 1e53                # Typical value for binding energy
fm3_to_km3          = (r_sun*1e18)**3     # from 1/fm^3 to 1/km^3 in solar masses


class TOV:
    """
    Class for the solution of the Tov equations
    all the quantities MUST BE in the following units:

    1) energy density [MeV/fm^3]
    2) pressure [MeV/fm^3]
    3) baryonic density [1/fm^3]

    The code converts The EoS in dimensionless units
    to do the calculation.
    """

    def __init__(self, ene, pre, nb_dens):
        """
        It converts The EoS in dimensionless units
        The EoS is interpolated with a cubic spline

        Parameters
        ---------
        ene : 1d array
            energy density [MeV/fm^3]
        pre : 1d array
            pressure [MeV/fm^3]
        nb_dens : 1d array
            baryonic density [1/fm^3]
        """
        # Convertion
        ene *= ene_constant1
        pre *= pre_constant2
        #Interpolation to get the energy density as a function of pressure
        self.rho = CubicSpline(pre, ene)#interp1d(pre, ene, kind='cubic')
        #Interpolation to get pressure as a function of the energy density
        self.press = CubicSpline(ene, pre)#interp1d(ene, pre, kind='cubic')
        #Interpolation to get pressure as a function of the baryonic density
        self.nb = CubicSpline(pre, nb_dens)#interp1d(pre, nb_dens, kind='cubic')

        #eos letta da file in unità cgs
        self.__rho = ene
        self.__pre = pre

        # Range of validity for the EoS
        self.min_rho = np.min(ene)
        self.max_rho = np.max(ene)

        self.min_p = np.min(pre)
        self.max_p = np.max(pre)


    def plot_eos(self):
        """
        Make the plot of the equation of state
        and the associated interpolation
        """
        conv1 = ene_constant1
        conv2 = pre_constant2

        # In the plot we convert again to obtain the
        #same unit in the EOS file
       

        fig1 = plt.figure(1)
        frame1=fig1.add_axes((.1,.35,.8,.6))
        #plot eos
        plt.plot(self.__rho/conv1, self.__pre/conv2, label='EOS', color='r')
        #plot interpolation
        plt.plot(self.__rho/conv1, self.press(self.__rho)/conv2, label='EOS_int', color='b')
        plt.ylabel(r'$P [MeV/fm^{3}]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid()


        #To check the interpolation
        frame3=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(self.__rho/conv1, (self.press(self.__rho)-self.__pre)/conv2, marker='.', linestyle='')
        plt.xlabel(r'$\varepsilon [MeV/fm^{3}]$')
        plt.ylabel('EOS-EOS_int')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()

        plt.show()


    def tov_eq(self, r, P, m):
        """
        Equation to solve

        Parameters
        ----------
        r : float
            position inside the star
        P : float
            Pressure at r
        M : float
            Mass at r

        Returns
        -------
        dPdr : float
            Pressure derivative computed in r
        dmdr : float
            Mass derivative computed in r
        """

        # The simulation can be done only in the range of validity for the EoS
        if P < self.min_p or P > self.max_p:
            return 0., 0.

        rho = self.rho(P)

        dPdr = -(rho + P)*(m + 4.0*np.pi*r**3*P)
        dPdr = dPdr/(r*(r - 2.0*m))

        dmdr = 4.0*np.pi*rho*r**2.

        return dPdr, dmdr


    def check_values(self, dens):
        """
        Checks whether the central density is in
        the range of validity for the EoS.
        Otherwise an exception is raised

        Parameter
        ---------
        dens : float
            density dimensionless variable
        """
        conv = ene_constant1
        if dens < self.min_rho or dens > self.max_rho:
            a = f'The central density chosen: {dens/conv:.5e} is out of range for the EOS.\n'
            b = f'the value must be between :{self.min_rho/conv:.5e} and {self.max_rho/conv:.5e}'
            errore = f'{a}{b}'
            raise Exception(errore)


    def RK4(self, f, P_c, M_c, dr):
        '''
        Computes the solution with runge_kutta4.

        Parameters
        ----------
        f : callable
            Function to integrate
        P_c : float
            Central pressure.
        M_c : float
            Central mass.
        dr : float
            Integration step.

        Returns
        -------
        r : 1d array
            radial coordinate of the star.
        P : 1d array
            Pressure profile
        M : 1d array
            Mass profile
        '''

        P = []
        M = []
        r = []

        P.append(P_c)
        M.append(M_c)
        r.append(dr)

        while P[-1] >= self.min_p:
            pk1, mk1 = f(r[-1], P[-1], M[-1])
            pk2, mk2 = f(r[-1] + dr/2, P[-1] + pk1*dr/2, M[-1] + mk1*dr/2)
            pk3, mk3 = f(r[-1] + dr/2, P[-1] + pk2*dr/2, M[-1] + mk2*dr/2)
            pk4, mk4 = f(r[-1] + dr, P[-1] + pk3*dr, M[-1] + mk3*dr)

            P_n = P[-1] + (dr/6)*(pk1 + 2*pk2 + 2*pk3 + pk4)
            M_n = M[-1] + (dr/6)*(mk1 + 2*mk2 + 2*mk3 + mk4)
            r_n = r[-1] + dr

            P.append(P_n)
            M.append(M_n)
            r.append(r_n)

        if P[-1] <= self.min_p:
            # to avoid an overshooting in the last step
            r = np.array(r[:-1])
            P = np.array(P[:-1])
            M = np.array(M[:-1])

        else :
            r = np.array(r)
            P = np.array(P)
            M = np.array(M)

        return r, P, M


    def solve(self, dens_c, dr):
        """
        Computes initial condition and solves the structure

        Parameters
        ---------
        dens_c : float
            central density [MeV/fm^3]
        dr : float
            integration step

        Return
        ------
        r : 1d array
            radial coordinate of the star in km
        P : 1d array
            pressure profile in 10^36 dyne/cm^2 .
        M : 1d array
            Mass profile in solar masses.
        M_grav : float
            gravitational mass of the star (the last value of M) in solar masses
        prop_mass : float
            proper mass ofthe star
            \int_0^R 4*pi*rho*r^2/sqrt(1-2M/r) dr   in solar masses
        bar_mass : float
            total mass of all the components of the star
            \int_0^R 4*pi*nb*r^2/sqrt(1-2M/r) dr in solar masses
        B_grav_ene : float
            Gravitational Binding Energy in units of 10^53 ergs
        B_int_ene : float
            Internal Binding Energy in units of 10^53 ergs
        tot_ene : float
            Total Binding Rnergy in units of 10^53 ergs
        """


        dr = dr/r_sun
        dens_c *= ene_constant1
        #check
        self.check_values(dens_c)
        #initial condition
        p = self.press(dens_c)
        rho = self.rho(p)
        m = (4/3)*np.pi*rho*dr**3.

        r, P, M = self.RK4(self.tov_eq, p, m, dr)
        M_grav = M[-1]

        #=====================================================
        # Proper Mass
        #=====================================================
        rho1 = self.rho(P)
        int1 = 4*np.pi*r**2*rho1/np.sqrt(1 - 2*M/r)
        prop_mass = integrate.simpson(int1, x=r)

        #=====================================================
        # Baryonic Mass
        #=====================================================
        rho2 = self.nb(P)
        rho2 = rho2*fm3_to_km3
        int2 = 4*np.pi*r**2*rho2/np.sqrt(1 - 2*M/r)
        bar_num = integrate.simpson(int2, x=r)
        bar_mass = bar_num * uma * MeV_to_g / M_sun

        #=====================================================
        # Gravitational Energy: E_G = M_G - M_P
        #=====================================================
        grav_ene  = M_grav - prop_mass
        grav_ene *= M_sun * c**2 #ergs
        grav_ene /= typical_energy
        B_grav_ene = - grav_ene # Gravitational binding energy BE_G

        #=====================================================
        # Internal Energy: E_I = M_P - M_B
        #=====================================================
        int_ene  = prop_mass - bar_mass
        int_ene *= M_sun * c**2   #ergs
        int_ene /= typical_energy
        B_int_ene = - int_ene # Internal binding energy BE_I

        #=====================================================
        # Total binding energy: BE_T = BE_G + BE_I = M_B - M_G
        #=====================================================
        tot_ene  = bar_mass - M_grav
        tot_ene *= M_sun * c**2 # ergs
        tot_ene /= typical_energy

        P *= pn_to_1036_dyne_cm2 # prssure in 10^36 dyne/cm^2
        r *= r_sun               # radius in km

        return r, P, M, M_grav, prop_mass, bar_mass, B_grav_ene, B_int_ene, tot_ene



    def MvR(self, central_density, dr):
        """
        Compute mass-radius relation and all
        the parameters of every star

        Parameters
        ----------
        central_density : float
            central density [MeV/fm^3]
        dr : float
            integration step

        Return
        ------
        R : 1d array
            Stellar radius in km.
        M : 1d array
            gravitational mass of all the stars in solar masses
        P : 1d array
            proper mass of all the stars in solar masses
        bar_mass : 1d array
            total mass of all the components of all the stars in solar masses
        B_grav_ene : 1d array
            Gravitational Binding Energy of all the stars in units of 10^53 ergs
        B_int_ene : 1d array
            Internal Binding Energy of all the stars in units of 10^53 ergs
        tot_ene : 1d array
            Total Binding Energy of all the stars in units of 10^53 ergs
        """

        N = len(central_density)
        M = np.zeros(N)
        R = np.zeros(N)
        P = np.zeros(N)
        B = np.zeros(N)
        BE_G = np.zeros(N)
        BE_I = np.zeros(N)
        BE_T = np.zeros(N)


        for i, dens_c in enumerate(central_density):
            r, p, m, M_grav, prop_mass, bar_mass, B_grav_ene, B_int_ene, tot_ene = self.solve(dens_c, dr)
            M[i] = M_grav
            R[i] = r[-2] + (r[-1] - r[-2])/(p[-1] - p[-2]) * (0 - p[-2])
            P[i] = prop_mass
            B[i] = bar_mass
            BE_G[i] = B_grav_ene
            BE_I[i] = B_int_ene
            BE_T[i] = tot_ene

            print(f"{(i+1)/N * 100:.1f} % \r", end='')


        return R, M, P, B, BE_G, BE_I, BE_T

