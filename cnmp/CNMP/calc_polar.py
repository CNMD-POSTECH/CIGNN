import numpy as np
from itertools import product
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor

def calc_polarization(structure, 
                      charge,
                      ase=False,
                      o_radius=0.25,
                      hf_radius=1.5,
                      unit_change=True,
                      return_center=False):
    
    if ase:
        structure = AseAtomsAdaptor.get_structure(structure)
    
    # Convert species and fractional coordinates to arrays
    species_array = np.array(structure.species)
    frac_coords_array = np.array(structure.frac_coords)
    charges_array = np.array(charge)

    # Identify indices for Hf and O atoms
    hf_indices = np.where(species_array == Element('Hf'))[0]
    o_indices = np.where(species_array == Element('O'))[0]

    # Prepare offsets for Hf and O atoms
    hf_offsets = np.array(list(product(range(-1, 2), repeat=3)))
    o_offsets = np.array(list(product(range(-1, 2), repeat=3)))

    # Initialize offsets for all species with appropriate size
    offsets = np.zeros((len(species_array), max(hf_offsets.shape[0], o_offsets.shape[0]), 3))
    offsets[hf_indices, :hf_offsets.shape[0], :] = hf_offsets
    offsets[o_indices, :o_offsets.shape[0], :] = o_offsets

    # Adjust fractional coordinates with offsets
    expanded_frac_coords = frac_coords_array[:, np.newaxis, :] + offsets
    expanded_cart_coords = structure.lattice.get_cartesian_coords(expanded_frac_coords.reshape(-1, 3)).reshape(expanded_frac_coords.shape)

    # Define lattice boundaries and margins for Hf and O atoms
    lattice_bounds = np.array([structure.lattice.a, structure.lattice.b, structure.lattice.c])
    hf_margin = np.array([-hf_radius, hf_radius])
    o_margin = np.array([-o_radius, o_radius])

    # Create masks for Hf and O atoms based on their respective margins
    hf_valid_mask = np.all((expanded_cart_coords >= hf_margin[0]) & (expanded_cart_coords <= lattice_bounds + hf_margin[1]), axis=2)
    o_valid_mask = np.all((expanded_cart_coords >= o_margin[0]) & (expanded_cart_coords <= lattice_bounds + o_margin[1]), axis=2)

    # Combine masks according to species
    valid_mask = np.zeros_like(hf_valid_mask)
    valid_mask[hf_indices] = hf_valid_mask[hf_indices]
    valid_mask[o_indices] = o_valid_mask[o_indices]

    # Collect valid coordinates and corresponding species, charges
    valid_cart_coords = expanded_cart_coords[valid_mask]

    # Calculate mean coordinates for valid sites
    unique_sites, inverse_indices = np.unique(np.repeat(np.arange(len(species_array)), offsets.shape[1])[valid_mask.ravel()], return_inverse=True)
    mean_coords = np.add.reduceat(valid_cart_coords, np.r_[0, np.flatnonzero(np.diff(inverse_indices)) + 1]) / np.bincount(inverse_indices)[:, np.newaxis]

    # Calculate Bader properties
    bader_x = charges_array[unique_sites] * mean_coords[:, 0]
    bader_y = charges_array[unique_sites] * mean_coords[:, 1]
    bader_z = charges_array[unique_sites] * mean_coords[:, 2]

    # Calculate polarization components
    # polarization_factor = 1602 / structure.volume (change uC/cm^2 unit)
    polarization_factor = 1
    sum_x = np.sum(bader_x) * polarization_factor
    sum_y = np.sum(bader_y) * polarization_factor
    sum_z = np.sum(bader_z) * polarization_factor
    
    # Return center positions if required
    if return_center:
        center_x = mean_coords[:, 0]
        center_y = mean_coords[:, 1]
        center_z = mean_coords[:, 2]
        if unit_change:
            return sum_x*1602/structure.volume, sum_y*1602/structure.volume, sum_z*1602/structure.volume, center_x, center_y, center_z
        else:
            return sum_x, sum_y, sum_z, center_x, center_y, center_z

    if unit_change:
        return sum_x*1602/structure.volume, sum_y*1602/structure.volume, sum_z*1602/structure.volume
    else:
        return sum_x, sum_y, sum_z