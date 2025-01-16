from cignn.utils.multibody.multibody_crystal_graph_generator import *
from cignn.model.invariant_CNMD import *
from cignn.train.trainer import *
from pymatgen.core import Structure
from torch import Tensor, nn
from tqdm import tqdm

import torch.optim as optim
import numpy as np
import argparse
import torch
import json
import time
import yaml
import sys
import os

def load_model(predict_config, device):
    checkpoint_path=os.path.join('./', predict_config['path'], predict_config['save']['best_save'])
    checkpoint=torch.load(checkpoint_path, map_location=device)
    config=checkpoint['args']
    normalizers_data=checkpoint['normalizers']
    normalizers={target:Normalizer(torch.zeros(1)) for target in config['target_list']}
    scale_datas=checkpoint['scale_datas']
    
    model_args=argparse.Namespace(**checkpoint['args'])

    scale_method=model_args.data['scale_method']
    if scale_method=='system' :
        energy_mean=normalizers['energy'].mean
        force_rms=normalizers['force'].rms
        if 'static_e' in scale_datas.keys() and model_args.data['charge']==True :
            q_energy_mean=scale_datas['static_e']
        else:
            q_energy_mean=None
    elif scale_method=='species' :
        energy_mean=scale_datas['species_atomE']
        force_rms=normalizers['force'].rms
        if 'species_staticE_atom' in scale_datas.keys() and  model_args.data['charge']==True :
            q_energy_mean=scale_datas['species_staticE_atom']
        else:
            q_energy_mean=None
            
    species_force_rms=scale_datas['species_force_rms']
    atom_type=torch.Tensor(list(species_force_rms.keys())).reshape(-1,1).to(device=device)

    model=InvCNMD(
                    atom_type=atom_type,
                    element_len=config['model']['element_len'],
                    atom_fea_len=config['model']['atom_fea_len'],
                    nbr_fea_len=config['model']['nbr_fea_len'],
                    angle_fea_len=config['model']['angle_fea_len'],
                    h_fea_len=config['model']['h_fea_len'],
                    n_conv=config['model']['n_conv'],
                    n_h=config['model']['n_h'],
                    num_radial=config['model']['num_radial'],
                    num_spherical=config['model']['num_spherical'],
                    direct=config['direct'],
                    charge=config['data']['charge'],
                    cutoff=config['data']['radius'],
                    energy_mean=energy_mean,
                    species_force_rms=species_force_rms,
                    force_rms=force_rms,
                    q_energy_mean=q_energy_mean,
                    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    return model, model_args, normalizers

def predict(predict_config):
    if predict_config['device']['number']:
        device=torch.device(predict_config['device']['type']+f':{predict_config['device']['number']}')
    else:
        device=torch.device(predict_config['device']['type'])

    test_files = [os.path.join(predict_config['data']['site'], file) for file in predict_config['data']['test_data']]
    test_data = []

    for test_file in test_files:
        test_data.extend(np.load(test_file, allow_pickle=True))

    model, model_args, normalizers=load_model(predict_config, device)
    model.eval()

    mae_errors={'e': AverageMeter(), 'f': AverageMeter(), 'q': AverageMeter()}
    predictions=[]
    
    print("Starting Inference...")
    total_inference_time=0.0

    for i, _ in enumerate(test_data):
        struc=Structure(lattice=test_data[i]['lattice'],
                        species=test_data[i]['atom_number'],
                        coords=test_data[i]['coords'],
                        coords_are_cartesian=True)
        
        crystal_dataset=StructureData(
                                    structure_data=struc,
                                    data_type='data',
                                    device=device,
                                    atom_type=model_args.data['atom_type'],
                                    nbr_mode=model_args.data['nbr_mode'],
                                    max_num_nbr=model_args.data['max_nbr'],
                                    radius=model_args.data['radius'],
                                    small_angle=model_args.data['small_angle'], 
                                    angle_cutoff=model_args.data['angle_cutoff'],
                                    local_cutoff_type=model_args.data['local_cutoff_type'],
                                    charge=model_args.data['charge'],
                                    q_model=predict_config['data']['q_model'],
                                    e_field=predict_config['data']['e_field']
                                    )

        data=crystal_dataset.get_data(device=device)
        
        torch.cuda.synchronize()  # GPU 연산 동기화
        start_time=time.time()
        
        output=model(data)
        
        torch.cuda.synchronize()  # GPU 연산 동기화
        end_time=time.time()

        if predict_config['prediction']['energy']:
            pred_E=normalizers['force'].descale(output['energy']).detach().cpu().numpy()[0][0]
            target_E=test_data[i]['E']/(data['position'].shape[0])
            energy_error=np.abs(pred_E - target_E)
            mae_errors['e'].update(energy_error, 1)
            tmp={
                'graph_id': i,
                'energy': {
                    'ground_truth': target_E,
                    'prediction': pred_E,
                },
            }

        if predict_config['prediction']['force']:
            pred_F=normalizers['force'].descale(output['force']).detach().cpu().numpy()
            target_F=test_data[i]['F']
            force_error=np.mean(np.abs(pred_F - target_F))
            mae_errors['f'].update(force_error, 1)
            tmp['forces']={
                'ground_truth': target_F,
                'prediction': pred_F,
            }

        if predict_config['prediction']['charge']:
            target_Q=np.array(test_data[i]['Q'])
            pred_Q=data['charge']
            if predict_config['data']['e_field']:
                pred_Q=data['field_charge']
            pred_Q=pred_Q[:, :1] if torch.all(pred_Q[:, 1:]==pred_Q[:, :-1], dim=1).all() else pred_Q
            pred_Q=pred_Q.squeeze()
            pred_Q=pred_Q.detach().cpu().numpy()
            charge_error=np.mean(np.abs(pred_Q - target_Q))
            mae_errors['q'].update(charge_error, 1)
            tmp['charge']={
                'ground_truth': target_Q,
                'prediction': pred_Q
            }

        inference_time=end_time - start_time
        total_inference_time+=inference_time
        predictions.append(tmp)

    save_path=os.path.join('./', predict_config['path'])
    np.save(f'{save_path}/pred.npy', predictions)

    # 평균 추론 시간 계산
    print('Length of test datas:', len(predictions))
    avg_inference_time=total_inference_time / len(predictions)
    print(f"Total inference time for batches: {total_inference_time:.4f}s")
    print(f"Average inference time per batch: {avg_inference_time:.4f}s")
    print(f"Energy MAE: {mae_errors['e'].avg:.4f}")
    print(f"Force MAE: {mae_errors['f'].avg:.4f}")
    print(f"Charge MAE: {mae_errors['q'].avg:.4f}")

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config={}
    return config

def main():
    parser=argparse.ArgumentParser(description="Run predict from config YAML")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args=parser.parse_args()

    config=load_config(args.config)
    predict(config)