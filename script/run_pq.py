######################################################################################
NPY_PATH = '/home/work/gyrudgyrud21/CIGNN/cnmp/example/pq.npy'
CIGNN_PATH = '/home/work/gyrudgyrud21/CIGNN'
ATOMS_PATH = '/home/work/gyrudgyrud21/CIGNN/cnmp/example/example.data'
######################################################################################

import sys
sys.path.append(f'{CIGNN_PATH}/cignn/utils')

from etc import *
from ase.io import read, write
from ase.build.tools import sort

atoms = read(ATOMS_PATH, format='lammps-data', atom_style='charge', sort_by_id=True)
atomtypes = {1: 'Hf', 2: 'O', 72: 'Hf', 8: 'O'}
for atom in atoms:
    atom.symbol = atomtypes[atom.number]
atoms = sort(atoms, tags=atoms.arrays['type'])
charge = atoms.get_initial_charges()
print(charge)
compute_amorphous_polarization(atoms,                       # ase atom file
                               charge,                      # should be array with shape n_atoms
                               ase=True, 
                               unit_change=False,
                               e_field=False, 
                               shift_value_path=NPY_PATH)