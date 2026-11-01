# Isentropic

Those codes solves the equations of the relativistic mean field model. The calculation are done at constant entropy per baryon S/A = const.
For simplicity, readibility and optimization of the code we dont adopt the same strategy of the Complete_T direcytory, we decide instead to
write a single code for each matter composition.

The iso_s.py file consider neutron, protron, electron and muon in beta equilibrium.
The iso_s_neutrinos.py file consider neutron, protron, electron and muon in beta equilibrium with neutrinos.
The iso_s_HYP.py file consider in addition all the other barion of the octet.
The iso_s_HYPn.py file is the same as iso_s_HYP.py but without neutrinos.
