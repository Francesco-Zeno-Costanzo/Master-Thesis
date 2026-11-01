# TOV

In this repository, we provide a simple code to solve the Tolman-Oppenheimer-Volkoff equations for a given equation of state.
TOV.py file is more general than the code TOV_p.py because now the EOS is something that we can choose and is an imput of the code.
The EOS must be in the form of a text file with three columns, each one with a specific quantity:

1) energy density [MeV/fm^3]
2) pressure [MeV/fm^3]
3) baryonic density [1/fm^3]

The output can be the structure of a single star or of a
sequence of stars to compute the mass-radius relation;
Futhermore the following quantities are computed for every star:

1) Gravitational mass [M_sun]
2) Proper        mass [M_sun]
3) Baryonic      mass [M_sun]
4) Gravitational binding energy [x 10^53 ergs]
5) Internal binding energy      [x 10^53 ergs]
6) Total binding energy         [x 10^53 ergs]
