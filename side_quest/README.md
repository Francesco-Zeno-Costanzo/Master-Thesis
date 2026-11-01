# Stellar-structure

Simple codes for the resolution of the stellar structure in hydrostatic equilibrium in the framework of the Newtonian gravity of general relativity, using a politropic equation of state.

## 1. Newtonian Stellar Structure

### 1.1 Hydrostatic Equilibrium

A spherically symmetric star in hydrostatic equilibrium is described by two coupled differential equations.

The conservation of mass gives:

$$
\frac{dm}{dr}=4\pi r^2\rho,
$$

where m(r) is the mass enclosed within radius r, and $\rho(r)$ is the mass density.
The balance between the pressure gradient and gravitational force gives

$$
\frac{dP}{dr}=-\frac{Gm\rho}{r^2},
$$

where P(r) is the pressure and G is the gravitational constant.

Together, these equations describe the internal structure of a Newtonian star.

### 1.2 Polytropic Equation of State

To close the system, an equation of state relating pressure and density is required.
The simplest possible choice is:

$$
P=K\rho^{1+1/n},
$$

where:

K is the polytropic constant;
n is the polytropic index;
$\rho$ is the mass density.

Different values of n describe different pressure–density relations. In particular, n=3/2 corresponds to a non-relativistic, completely degenerate ideal Fermi gas.

### 1.3 Lane–Emden Equation

Introducing the dimensionless density function \theta,

$$
\rho=\rho_c\theta^n,
$$

where $\rho_c$ is the central density, and defining a dimensionless radial coordinate, the Newtonian equations can be reduced to the Lane–Emden equation:

$$
\frac{1}{\xi^2}\frac{d}{d\xi}
\left(\xi^2\frac{d\theta}{d\xi}\right)
=-\theta^n.
$$

The central boundary conditions are:

$$
\theta(0)=1,
\qquad
\theta'(0)=0.
$$

The stellar surface is located at the first zero of $\theta$. Once this point is found, the stellar radius and total mass can be recovered from the corresponding dimensional scales.

## 2. Stellar Structure in General Relativisty

For sufficiently compact objects, Newtonian gravity is no longer adequate. General Relativity describes hydrostatic equilibrium through the Tolman–Oppenheimer–Volkoff equations.

### 2.1 TOV Equations

Let $\epsilon(r)$ denote the total energy density, including rest-mass energy and internal energy, expressed in $\mathrm{J,m^{-3}}$. Let m(r) be the gravitational mass enclosed within radius r.

The TOV equations are:

$$
\frac{dm}{dr}
=\frac{4\pi r^2}{c^2}\epsilon,
$$

and

$$
\frac{dP}{dr}=
-\frac{G(\epsilon+P)}{c^2r^2}
\frac{
m+\dfrac{4\pi r^3P}{c^2}
}{
1-\dfrac{2Gm}{rc^2}
}.
$$

Here, c is the speed of light.

### 2.2 Polytropic Equation of State

We always use a polytropic equation of state. Rather than integrating the dimensional equations directly, the TOV equations are rewritten in dimensionless form. This approach is analogous to the one used for the Lane–Emden equation.
