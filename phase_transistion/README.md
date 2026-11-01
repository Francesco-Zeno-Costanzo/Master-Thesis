# phase_transition

In this repository, we provide the code used to compute the phase transition between liquid and gas phase in nuclear matter. The code is based on the relativistic mean field model with density-dependent coupling constants. The code is divided into different files, each one with a specific task, to achieve a simple organization of the code.

In cut_off.py we compute the minimum of energy per nucleon in symmetric matter with the sigma-omega model in relativistic mean field at finite T (/= 0)
to the variation of the integration cut_off to see if there is stability and choose the integration limit, which ideally would be infinite.

The idea is that we want to find at which T the pressure has an inflection with horizontal tangent then we compute the derivative and
we find all the minima of the derivative. After this we find the zeros of the minima of the derivative as a function of T
because the derivative of a function that has inflection with horizontal tangent must have minimum equal to zero.
