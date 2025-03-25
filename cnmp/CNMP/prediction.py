import os
import sys
import math
import yaml
import torch
import warnings
import numpy as np
from ase import Atoms
from time import time
from ase.io import read, write

from generator import AtomsData
from cignn.utils.multibody.multibody_crystal_graph_generator import Normalizer
from cignn.model.invariant_CNMD import InvCNMD

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)

def CNMP_initialize(gpu=True, gpu_name='cuda:0', e_field_list=False):
    log_file = open('log.CNMP', 'w')
    log_file.write('\n')

    gpu_ = (gpu and torch.cuda.is_available())
    map_location = gpu_name if gpu_ else "cpu"
    log_file.write("gpu (in)   = " + str(gpu) + "\n")
    log_file.write("gpu (eff)  = " + str(gpu_) + "\n")
    log_file.write(f"gpu (number) = {map_location}\n")

    basePath = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.normpath(os.path.join(basePath, 'config'))
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    if not yaml_files:
        raise FileNotFoundError("No .yaml files found in the 'config' directory.")
    config_yml = os.path.normpath(os.path.join(config_dir, yaml_files[0]))
    with open(config_yml, 'r') as file:
        config = yaml.safe_load(file)

    chekpt_dir = os.path.normpath(os.path.join(basePath, 'checkpoint'))
    pth_files = [f for f in os.listdir(chekpt_dir) if f.endswith('.pth.tar')]
    if not pth_files:
        raise FileNotFoundError("No .pth.tar files found in the 'checkpoint' directory.")
    checkpoint = os.path.normpath(os.path.join(chekpt_dir, pth_files[0]))
    if 'data' not in config:
        raise KeyError("Key 'data' not found in the configuration file.")
    qeq_checkpoint = config['data']['q_model']

    log_file.write("config_yml = " + config_yml + "\n")
    log_file.write("checkpoint = " + checkpoint + "\n")
    log_file.write("qeq_checkpoint = " + str(qeq_checkpoint) + "\n")

    max_nbr = config['data'].get('max_nbr', 16)
    cutoff = config['data'].get('cutoff', 5.0)

    log_file.write("max_neigh  = " + str(max_nbr) + "\n")
    log_file.write("cutoff     = " + str(cutoff) + "\n")

    assert max_nbr > 0
    assert cutoff > 0.0

    model_data = torch.load(checkpoint, map_location=map_location)

    global myNorm

    myNorm = {}

    normalizers_datas = model_data['normalizers']
    for target in normalizers_datas.keys():
        normalizer = Normalizer(torch.zeros(1, device=map_location))
        normalizer.load_state_dict(normalizers_datas[target])
        myNorm[target] = normalizer
        
    force_rms = myNorm['force'].rms.cpu()

    scale_datas = model_data['scale_datas']
    energy_mean = scale_datas['species_atomE']
    species_force_rms = scale_datas['species_force_rms']
    q_energy_mean = scale_datas['species_staticE_atom']
    log_file.write('energy_mean = ' + str(energy_mean) + '\n')
    log_file.write('species_force_rms = ' + str(species_force_rms) + '\n')

    atom_type = torch.tensor(list(species_force_rms.keys()), dtype=torch.float64).reshape(-1, 1).to(map_location)
    log_file.write('atom_type = ' + str(atom_type) + '\n')

    global myModel

    myModel = InvCNMD(
                    atom_type=atom_type,
                    element_len=model_data['args']['model']['element_len'],
                    atom_fea_len=model_data['args']['model']['atom_fea_len'],
                    nbr_fea_len=model_data['args']['model']['nbr_fea_len'],
                    angle_fea_len=model_data['args']['model']['angle_fea_len'],
                    h_fea_len=model_data['args']['model']['h_fea_len'],
                    n_conv=model_data['args']['model']['n_conv'],
                    n_h=model_data['args']['model']['n_h'],
                    num_radial=model_data['args']['model']['num_radial'],
                    num_spherical=model_data['args']['model']['num_spherical'],
                    direct=model_data['args']['direct'],
                    charge=model_data['args']['data']['charge'],
                    cutoff=model_data['args']['data']['radius'],
                    energy_mean=energy_mean,
                    force_rms=force_rms,
                    species_force_rms=species_force_rms,
                    q_energy_mean=q_energy_mean)

    try:
        myModel.load_state_dict(model_data['state_dict'])
    except:
        print('Check your Model')

    myModel.to(map_location)
    myModel.eval()

    global myAtoms
    myAtoms = None

    global e_field_bool
    e_field_bool = False
    if e_field_list is not None and not (math.isclose(e_field_list[0], 0) and math.isclose(e_field_list[1], 0)):
        log_file.write('E-field = ' + str(e_field_list) + '\n')
        e_field_bool = True

    global myA2G

    myA2G = AtomsData(
                    nbr_mode=config['data']['nbr_mode'],
                    max_num_nbr=config['data']['max_nbr'],
                    radius=config['data']['radius'],
                    small_angle=config['data']['small_angle'],
                    angle_cutoff=config['data']['angle_cutoff'],
                    local_cutoff_type=config['data']['local_cutoff_type'],
                    charge=model_data['args']['data']['charge'],
                    q_model=qeq_checkpoint,
                    e_field=e_field_list if e_field_bool else False,
                    device=map_location,
                    atom_type=atom_type)

    global e_field_params
    e_field_params = e_field_list

    return cutoff

def CNMP_get_energy_forces_and_charge(cell, atomic_numbers, positions):
    global myAtoms

    if myAtoms is None or len(myAtoms.numbers) != len(atomic_numbers):
        myAtoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=[True, True, True]
        )
    else:
        myAtoms.set_cell(cell)
        myAtoms.set_atomic_numbers(atomic_numbers)
        myAtoms.set_positions(positions)

    global myA2G
    graph_data = myA2G.get_data(myAtoms)
    if e_field_bool:
        charge_data = graph_data['field_charge']
        chi = graph_data['field_chi']
    else:
        charge_data = graph_data['charge']
        chi = graph_data['chi']

    # Use PyTorch's built-in functions to simplify logic
    atomic_charge = charge_data[:, :1] if torch.all(charge_data[:, 1:] == charge_data[:, :-1], dim=1).all() else charge_data
    chi = chi[:, :1] if torch.all(chi[:, 1:] == chi[:, :-1], dim=1).all() else chi

    atomic_charge = atomic_charge.squeeze()
    chi = chi.squeeze()

    global myModel
    predictions = myModel(graph_data)

    global myNorm
    scale_factor = myNorm['force']
    energy = scale_factor.descale(predictions['energy']).detach().cpu().numpy().item() * len(atomic_numbers)
    forces = scale_factor.descale(predictions['force']).detach().cpu().numpy()
    charge = atomic_charge.detach().cpu().numpy()
    chi = chi.detach().cpu().numpy()

    if e_field_bool:
        e_field_s, e_field_e, axis = e_field_params

        if e_field_s == e_field_e:
            E = e_field_s
            add_force = (E * charge).reshape(-1, 1)
            forces[:, axis] += add_force.squeeze()
        else:
            print('Not implemented yet')

    energy_field = energy

    return energy, energy_field, forces.tolist(), charge.tolist(), chi.tolist()