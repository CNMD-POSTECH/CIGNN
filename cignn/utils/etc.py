# ─────────────────────────────────────────────────────────────────────────────
# Author: Hyo Gyeong Shin
# Affiliation: CNMD
# Description: Custom polarization computation and site replacement script for structure analysis using charges. 

# Written entirely by Hyo Gyeong Shin.
# ─────────────────────────────────────────────────────────────────────────────

import random
import numpy as np
import pandas as pd
from itertools import product
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CutOffDictNN

# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_amorphous_polarization(struct: Structure, charges: np.ndarray,
                                   hf_radius=0.8, o_radius=0.1, ase=False,
                                   unit_change=True, return_center=False,
                                   e_field=False, shift_value_path='pq.npy'):
    """
    Compute total polarization (Px, Py, Pz) for an HfO2 structure
    using the fractional‐wrap + bader charge approach.

    struct: Lattice and fractional coordinates of the crystal.
    charges: 1-D array of Bader charges aligned with `struct.sites`.
    hf_radius, o_radius: Real-space margins (Å) used to decide whether a periodic image is still “inside” the augmented unit cell.
    unit_change: If True, convert from e·Å to μC cm⁻² (factor 1602)
    return_center: If True, also return the (x, y, z) coordinates of the charge centroids.~
    """
    if ase:
        struct= AseAtomsAdaptor.get_structure(struct)

    # --- 1 │ identify atomic indices by species --------------------------------
    species_array = np.array(struct.species)
    hf_indices = np.where(species_array == Element('Hf'))[0]
    o_indices = np.where(species_array == Element('O'))[0]
    
    group_size = int(len(o_indices)/2)

    # --- 2 │ generate 3×3×3 periodic images via fractional offsets -------------
    hf_offsets = np.array(list(product(range(-1, 2), repeat=3)))
    o_offsets = hf_offsets.copy()

    offsets = np.zeros((len(species_array), max(hf_offsets.shape[0], o_offsets.shape[0]), 3))
    offsets[hf_indices, :hf_offsets.shape[0], :] = hf_offsets
    offsets[o_indices, :o_offsets.shape[0], :] = o_offsets

    frac_coords_array = np.array(struct.frac_coords)
    expanded_frac_coords = frac_coords_array[:, np.newaxis, :] + offsets
    expanded_cart_coords = struct.lattice.get_cartesian_coords(expanded_frac_coords.reshape(-1, 3)).reshape(expanded_frac_coords.shape)

    vec_lengths = np.array([struct.lattice.a, struct.lattice.b, struct.lattice.c])

    # --- 3 │ build species-dependent margin in fractional units ---------------
    hf_margin_frac = hf_radius / vec_lengths
    o_margin_frac  = o_radius  / vec_lengths

    exp_frac = expanded_frac_coords

    hf_valid_mask = np.all((exp_frac >= -hf_margin_frac) & (exp_frac <= 1 + hf_margin_frac), axis=2)
    o_valid_mask = np.all((exp_frac >= -o_margin_frac) & (exp_frac <= 1 + o_margin_frac), axis=2)
    
    #print("\n[ Hf Valid Image Offsets ]")
    #for idx in hf_indices:
    #    for i, valid in enumerate(hf_valid_mask[idx]):
    #        if valid:
    #            print(f"Hf atom index {idx}: offset = {hf_offsets[i]}")

    # Print O image offsets that are considered valid
    #print("\n[ O Valid Image Offsets ]")
    #for idx in o_indices:
    #    for i, valid in enumerate(o_valid_mask[idx]):
    #        if valid:
    #            print(f"O atom index {idx}: offset = {o_offsets[i]}")

    valid_mask = np.zeros_like(hf_valid_mask)
    valid_mask[hf_indices] = hf_valid_mask[hf_indices]
    valid_mask[o_indices] = o_valid_mask[o_indices]

     # --- 4 │ coordinates of all *valid* images ---------------------------------
    valid_cart_coords = expanded_cart_coords[valid_mask]

    unique_sites, inverse_indices = np.unique(np.repeat(np.arange(len(species_array)), offsets.shape[1])[valid_mask.ravel()], return_inverse=True)
    mean_coords = np.add.reduceat(valid_cart_coords, np.r_[0, np.flatnonzero(np.diff(inverse_indices)) + 1]) / np.bincount(inverse_indices)[:, np.newaxis]

    # --- 5 │ accumulate charge moments -----------------------------------------
    bader_x = charges[unique_sites] * mean_coords[:, 0]
    bader_y = charges[unique_sites] * mean_coords[:, 1]
    bader_z = charges[unique_sites] * mean_coords[:, 2]

    sum_x = np.sum(bader_x)
    sum_y = np.sum(bader_y)
    sum_z = np.sum(bader_z)

     # --- 6 │ unit conversion ----------------------------------------------------
    if unit_change:
        sum_x = sum_x*1602/struct.volume
        sum_y = sum_y*1602/struct.volume
        sum_z = sum_z*1602/struct.volume

    polarization = np.array([sum_x, sum_y, sum_z])
    
    if e_field:
        # shift_value는 polarization_shift - polarization_no_shift
        shift_value = np.load(f"{shift_value_path}")
        polarization = polarization + shift_value
    else:
        # from allegro-pol paper
        lattice_matrix = struct.lattice.matrix
        a_len = np.linalg.norm(lattice_matrix[0])
        b_len = np.linalg.norm(lattice_matrix[1])
        c_len = np.linalg.norm(lattice_matrix[2])
        if unit_change:
            polarization = np.array([polarization[0]*struct.volume/1602, polarization[1]*struct.volume/1602, polarization[2]*struct.volume/1602])
        else:
            polarization = np.array([polarization[0], polarization[1], polarization[2]])
        pq = np.array([a_len, b_len, c_len])
        g = lattice_matrix/np.linalg.norm(lattice_matrix, axis=1, keepdims=True) # 방향 벡터
        polarization_frac = np.dot(g, polarization)                             # 분극을 각 축에 대해 projection 시킨 후 polarizatio_frac 추출
        polarization_new = polarization_frac % (np.sign(polarization_frac)*pq)  # %p, branch 모호성 제거 (정수배 잘라냄), 아래에서 [-0.5*p와 0.5*p로 최종 값 이동]
        polarization_new = np.where(polarization_new>0.5*pq, polarization_new-pq, polarization_new)
        polarization_new = np.where(polarization_new<-0.5*pq, polarization_new+pq, polarization_new)
        polarization_new_cart = np.linalg.inv(g) @ polarization_new             # projection 해제한 분극 값
        if unit_change:
            polarization_final = polarization_new_cart/struct.volume * 1602
            shift_value = polarization_final - (polarization/struct.volume * 1602)
        else:
            polarization_final = polarization_new_cart
            shift_value = polarization_final - (polarization)            
        np.save(f"{shift_value_path}", shift_value)
        polarization = polarization_final
        #print(shift_value_path, shift_value)

    polarization[np.abs(polarization) < 0.1] = 0.0

    if return_center:
        center_x = mean_coords[:, 0]
        center_y = mean_coords[:, 1]
        center_z = mean_coords[:, 2]
        return polarization, center_x, center_y, center_z
    else:
        return polarization