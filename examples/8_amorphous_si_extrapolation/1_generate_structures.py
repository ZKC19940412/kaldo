from ase.optimize import LBFGSLineSearch
from ase.io import read,write
from mpi4py import MPI
import numpy as np
import subprocess
import lammps
import os
random_seed = 7793

def change_concentration(ge_concentration=0):
	# Read in 1728 atom system
	atoms = read('structures/1728_atom/aSi.xyz', format='xyz')

	# Swap in Ge atoms
	if ge_concentration != 0:
		symbols = np.array(atoms.get_chemical_symbols())
		n_ge_atoms = int(np.round(ge_concentration * len(symbols), 0))
		rng = np.random.default_rng(seed=random_seed)
		id = rng.choice(len(atoms), size=n_ge_atoms, replace=False)
		symbols[id] = 'Ge'
		atoms.set_chemical_symbols(symbols.tolist())
	ge_concentration = str(int(ge_concentration*100))
	folder_string = 'structures/1728_atom/aSiGe_C'+str(ge_concentration)
	if not os.path.exists(folder_string):
		os.makedirs(folder_string)

	# Minimize structure - LAMMPS + ASE
	lammps_inputs = {'lmpcmds': ["pair_style tersoff",
				"pair_coeff * * forcefields/SiCGe.tersoff Si(D) Ge"],
				"log_file" : "min.log",
				"keep_alive":True}
	calc = LAMMPSlib(**lammps_inputs)
	atoms.set_calculator(calc)
	atoms.pbc = True
	search = LBFGSLineSearch(atoms)
	search.run(fmax=.001)
	write('structures/1728_atom/aSiGe_C'+str(ge_concentration)+'/replicated_atoms.xyz',
		 search.atoms, format='xyz')
	# write('structures/1728_atom/aSiGe'+str(ge_concentration)+'/atoms.lmp',
	# 	 search.atoms, format='lammps-data')

	# # Find Force Constants - LAMMPS
	# cmdargs = ['-var', 'conc', ge_concentration, '-log', 'dyn.log']
	# lmp = lammps.lammps(cmdargs=cmdargs)
	# lmp.file('in.lmp')
	# mpi = MPI.COMM_WORLD.Get_rank()
	# nprocs = MPI.COMM_WORLD.Get_size()
	# MPI.finalize()


desired_concentrations = [0.0, 0.1]
desired_concentrations = [0.1]
for c in desired_concentrations:
	change_concentration(c)

