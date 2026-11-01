"""
The code calculates the pressure as a function of the number
density for each temperature between Tmin and Tmax e
stores all data in several files that will be read for
find at what temperature the pressure has an inflection
with horizontal tangent
"""
import time
import numpy as np
import Baryon_finite_T as bft

start = time.time()

path1  = fr"data_pt"
path2  = fr"data_pt/parameters.txt"

Tmin = 12.5                             #minimum of temperature
Tmax = 13.5                             #maximum of temperature
Tnum = 20                               #number of temperatue
Ture = np.linspace(Tmin, Tmax, Tnum)    #array of temperatue
number_dens = 50                       #number of density
min_dens = 0.001                        #minimum of density
max_dens = 0.2                          #maximun of density

Ture = np.array([10, 11.5, 13])
for i, T in enumerate(Ture):
    start1 = time.time()

    #computation of pressure
    _, _, press, _, _, _, _, _, _, _, _ = bft.main(min_dens, max_dens, number_dens, T)

    #save on file
    nome = f'/dati{i}.npy'
    np.save(path1+nome, press)

    end1 = time.time() - start1

    print(f'{i}-th temperature completed in {end1:.3f} s')


end = time.time() - start

print(f'Elapsed time = {end:.3f} seconds')

file = open(path2, "w")
file.write(f"{number_dens} \n")
file.write(f"{min_dens} \n")
file.write(f"{max_dens} \n")

for T in Ture:
    file.write(f"{T} \n")

file.close()
