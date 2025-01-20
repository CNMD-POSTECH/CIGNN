import yaml
import wandb
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR,ReduceLROnPlateau

from cignn.utils.multibody.multibody_crystal_graph_generator import *
from cignn.train.trainer import ModelTrainer
from cignn.model.invariant_CNMD import *

torch.set_default_dtype(torch.float64)

def target_statistics(data_loader,charge,q_model,target_list,Normalizer):
    if 'charge' in target_list:
        target_list.remove('charge')   
    data_list={target:list() for target in target_list} #energy,force
    scale_target=[]
    scale_target.append('atom_number')
    scale_target.append('atom_type')
    if charge==True:
        scale_target.append('static_e')
        scale_target.append('static_e_atom')
    data_list2={target:list() for target in scale_target} #energy,force,atom_number,atom_type
    data_list.update(data_list2)
    
    atom_counts=[]
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        atom_counts.append(torch.unique(data['crystal_atom_idx'], return_counts=True)[1].reshape(-1, 1))
        for target in list(data_list.keys()):
            data_list[target].append(data[target])
    for target in list(data_list.keys()):
        data_list[target]=torch.concat(data_list[target])
    system_e=data_list['energy']
    atom_counts=torch.concat(atom_counts)    
    data_list['energy']=data_list['energy']/atom_counts
    
    normalizers={target:Normalizer(data_list[target]) for target in target_list }
    
    print('calculate scale datas')
    atom_type_data=data_list['atom_type']
    atom_type_data=atom_type_data.to(torch.double)
    non_zero_cols=atom_type_data.any(dim=0)
    atom_type_data=atom_type_data[:, non_zero_cols]
    atomic_e=torch.linalg.lstsq(atom_type_data,system_e)[0] # which i have to get

    atom_type=torch.unique(data_list['atom_number']).reshape(-1,1)
    species_force_rms_dic={}
    species_atomE_dic={}
    if charge==True:
        species_staticE_atom_dic={}
    for i in range(len(atom_type)):
        species_atomE_dic[atom_type[i]]=atomic_e[i]
    for i in range(len(atom_type)):
        species=atom_type[i]
        idx=torch.where(data_list['atom_number']==species)[0]
        species_force=data_list['force'][idx]
        species_force_rms=torch.sqrt(torch.mean(species_force**2))
        species_force_rms_dic[species]=species_force_rms
        if charge==True:
            species_staticE_atom=torch.mean(data_list['static_e_atom'][idx])
            species_staticE_atom_dic[species]=species_staticE_atom

    scale_datas={}
    scale_datas['species_force_rms']=species_force_rms_dic
    scale_datas['species_atomE']=species_atomE_dic
    if charge==True:
        scale_datas['species_staticE_atom']=species_staticE_atom_dic
        scale_datas['static_e']=(data_list['static_e']/atom_counts).mean()
    return (normalizers,scale_datas)
    
def train(config):
    warnings.filterwarnings("ignore")

    try:
        os.makedirs(config['path'])
    except:
        pass

    if config['wandb'] :
        parser=argparse.ArgumentParser(description='Train')
        os.makedirs(os.path.join(config['path'],'wandb_logs'), exist_ok=True)
        run=wandb.init(project=config['wandb_project'], name=config['wandb_name'], dir=os.path.join(config['path'],'wandb_logs'))
         
    if config['device']['number']:
        device=torch.device(f"cuda:{config['device']['number']}" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    do_test=config['test']

    print('start loading input datas')
    crystal_kwargs={
        "data_site": config['data']['site'],
        "device": device,
        "nbr_mode": config['data']['nbr_mode'],
        "max_num_nbr": config['data']['max_nbr'],
        "radius": config['data']['radius'],
        "small_angle": config['data']['small_angle'],
        "angle_cutoff": config['data']['angle_cutoff'],
        "local_cutoff_type": config['data']['local_cutoff_type'],
        "charge": config['data']['charge'],
        "q_model": config['data']['q_model'],
        "interval": config['data']['interval'],
        "atom_type": config['data']['atom_type']
    }
    
    normalizers_file=os.path.join(config['path'], 'normalizers.pkl')
    scale_data_file=os.path.join(config['path'], 'scale_datas.pkl')
    train_loader_file=os.path.join(config['path'], 'train_loader.pkl')
    valid_loader_file=os.path.join(config['path'], 'valid_loader.pkl')
    test_loader_file=os.path.join(config['path'], 'test_loader.pkl')

    if all(os.path.isfile(f) for f in [normalizers_file, scale_data_file, train_loader_file, valid_loader_file]):
        normalizers=load_data(normalizers_file)
        scale_datas=load_data(scale_data_file)
        train_loader=load_data(train_loader_file)
        valid_loader=load_data(valid_loader_file)
        test_loader=load_data(test_loader_file) if do_test else None
    
    else:
        if config['data']['preprocess']['bool']:
            crystal_kwargs["data_site"] = config['path']
            split_data(save_path=config['path'], data_config=config['data'], test=do_test)
            train_data=['split_train.npy']
            valid_data=['split_valid.npy']
            if do_test:
                test_data=['split_test.npy']
        else:
            train_data=config['data']['train_data']
            valid_data=config['data']['valid_data']
            if do_test:
                test_data=config['data']['test_data']
        train_dataset=CrystalDataCIF(data_list=train_data, **crystal_kwargs)
        valid_dataset=CrystalDataCIF(data_list=valid_data, **crystal_kwargs)
        test_dataset=False
        if do_test:
            test_dataset=CrystalDataCIF(data_list=test_data, **crystal_kwargs)
        
        print('finish loading input datas')
        collate_fn=collate_pool
        train_loader, valid_loader, test_loader=get_loader( train_dataset,
                                                            valid_dataset,
                                                            test_datset=test_dataset,
                                                            batch_size=config['batch'],
                                                            collate_fn=collate_fn,
                                                            num_workers=0,
                                                            pin_memory=True,
                                                            return_test=True)

        print('do normalizing')
        normalizers,scale_datas=target_statistics(data_loader=train_loader,
                                                  charge=config['data']['charge'],
                                                  target_list=['energy','force'],
                                                  q_model=config['data']['q_model'],
                                                  Normalizer=Normalizer)
        atom_type=torch.Tensor(list(scale_datas['species_force_rms'].keys())).reshape(-1,1).to(device=device)
        for i in normalizers:
            if i in config['target_list']:
                print(f'{i} : {normalizers[i].state_dict()}')
        print('finish normalizing')

        save_data(normalizers_file, normalizers)
        save_data(scale_data_file, scale_datas)
        save_data(train_loader_file, train_loader)
        save_data(valid_loader_file, valid_loader)
        if do_test:
            save_data(test_loader_file, test_loader)
        
    if config['resume'] !='None':
        print('load previous model data')
        model_data=torch.load(config['resume'],map_location=lambda storage, loc: storage)
        model_args=argparse.Namespace(**model_data['args'])
        normalizers_data=model_data['normalizers']
        normalizers={target:Normalizer(torch.zeros(1)) for target in config['target_list']}
        for target in config['target_list'] :
            normalizers[target].load_state_dict(state_dict=normalizers_data[target])
        scale_datas=model_data['scale_datas']
        atom_type=torch.Tensor(list(scale_datas['species_force_rms'].keys())).reshape(-1,1).to(device=device)
    else:
        atom_type = torch.Tensor(list(scale_datas['species_force_rms'].keys())).reshape(-1, 1).to(device=device)
        for i in normalizers:
            if i in config['target_list']:
                print(f'{i} : {normalizers[i].state_dict()}')

    scale_method=config['data']['scale_method']
    if scale_method=='system' :
        energy_mean=normalizers['energy'].mean
        force_rms=normalizers['force'].rms
        if 'static_e' in scale_datas.keys() and config['data']['charge']==True :
            q_energy_mean=scale_datas['static_e']
        else:
            q_energy_mean=None
    elif scale_method=='species' :
        energy_mean=scale_datas['species_atomE']
        force_rms=normalizers['force'].rms
        if 'species_staticE_atom' in scale_datas.keys() and config['data']['charge']==True :
            q_energy_mean=scale_datas['species_staticE_atom']
        else:
            q_energy_mean=None
    else :
        raise ValueError('check scale_method in config')
    
    species_force_rms=scale_datas['species_force_rms']
    if config['wandb'] :
        config['model']['q_energy_mean']=str(q_energy_mean)
        config['model']['energy_mean']=str(energy_mean)
        config['model']['force_rms']=str(force_rms)
        config['model']['species_force_rms']=str(species_force_rms)
        wandb.config.update(config)
    if config['resume'] !='None':
        model=InvCNMD(
                    atom_type=atom_type,
                    element_len=model_args.model['element_len'],
                    atom_fea_len=model_args.model['atom_fea_len'],
                    nbr_fea_len=model_args.model['nbr_fea_len'],
                    angle_fea_len=model_args.model['angle_fea_len'],
                    h_fea_len=model_args.model['h_fea_len'],
                    n_conv=model_args.model['n_conv'],
                    n_h=model_args.model['n_h'],
                    num_radial=model_args.model['num_radial'],
                    num_spherical=model_args.model['num_spherical'],
                    direct=model_args.direct,
                    charge=model_args.data['charge'],
                    cutoff=model_args.data['radius'],
                    energy_mean=energy_mean,
                    species_force_rms=species_force_rms,
                    force_rms=force_rms,
                    q_energy_mean=q_energy_mean
                    )
        model.load_state_dict(model_data['state_dict'])
    else:
        model=InvCNMD(
                    atom_type=atom_type,
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
                    element_len=config['model']['element_len']
                    )
    model.to(device)

    # choose optimizer
    if config['resume'] !='None':
        config['optimizer']['algo']==model_args.optimizer['algo']
    
    if config['optimizer']['algo']=='Adam':
        optimizer=optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'],amsgrad=True)
    elif config['optimizer']['algo']=='SGD':
        optimizer=optim.SGD(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['algo']=='NAdam':
        optimizer=optim.NAdam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['algo']=='AdamW':
        optimizer=optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'],amsgrad=True)
    elif config['optimizer']['algo']=='RAdam':
        optimizer=optim.RAdam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    else:
        raise ValueError('check optimizer algo in config.yaml')
    
    # choose scheduler
    if config['resume'] !='None':
        optimizer.load_state_dict(model_data['optimizer'])
    
    # choose scheduler
    if config['scheduler']['algo']=='ReduceLROnPlateau':
        scheduler=ReduceLROnPlateau(optimizer, 
                                    mode='min', 
                                    factor=config['scheduler']['gamma'], 
                                    patience=config['scheduler']['patience'], 
                                    verbose=True, 
                                    threshold=config['scheduler']['threshold'], 
                                    threshold_mode='rel', 
                                    cooldown=config['scheduler']['cooldown'], 
                                    min_lr=config['scheduler']['min_lr'], 
                                    eps=config['scheduler']['eps'])
    else:
        raise ValueError('check scheduler algo in config.yaml')
   
    target_list=config['target_list']
    criterion={x:nn.MSELoss() for x in target_list }
    
    trainer=ModelTrainer(
                        path=config['path'],
                        target_list=target_list,
                        model=model,
                        device=device,
                        criterion=criterion,
                        optimizer=optimizer,
                        rmse_loss=True,
                        normalizers=normalizers,
                        print_freq=config['print_freq']
                        )
        
    best_mae_error=config['loss']['best_mae_error']
    weight_f=config['loss']['weight_f']
    weight_e=config['loss']['weight_e']
    starter_learning_rate=config['optimizer']['lr']

    lr=optimizer.param_groups[0]['lr']

    if config['resume'] !='None':
        if config['start_epoch'] !='None':
            start_epoch=config['start_epoch']
        else:
            start_epoch=model_data['epoch']
        end_epoch=start_epoch+config['epochs']
    else:
        start_epoch=0
        end_epoch=config['epochs']

    for epoch in range(start_epoch, end_epoch):
        batch_time_t, losses_t, maes_t=trainer.train(loader=train_loader,
                                                     epoch=epoch, 
                                                     loss_weight_force=weight_f,
                                                     loss_weight_energy=weight_e)        # evaluate on validation set
        batch_time_v, losses_v, maes_v=trainer.validate(
                                                     loader=valid_loader, 
                                                     epoch=epoch,
                                                     direct=config['direct'], 
                                                     loss_weight_force=weight_f,
                                                     loss_weight_energy=weight_e)

        if scheduler is not None:
            if config['scheduler']['algo']=='ReduceLROnPlateau':
                scheduler.step(losses_v['total'].avg)
            else:
                scheduler.step()
        lr=optimizer.param_groups[0]['lr']
        lr_log={'lr':lr,
                'loss_weight_e':weight_e,
                'loss_weight_f':weight_f}

        # remember the best mae_eror and save checkpoint
        is_best=maes_v['total'].avg < best_mae_error

        if do_test:
            # do test if best or last epoch
            if is_best==True:
                batch_time_b, losses_b, maes_b=trainer.test(loader=test_loader, 
                                                            epoch=epoch,
                                                            direct=config['direct'],
                                                            loss_weight_force=weight_f,
                                                            loss_weight_energy=weight_e,
                                                            )
                test_losses={f'{x}_loss@test':losses_b[x].avg for x in losses_b.keys()}
                test_maes={f'{x}_mae@test':maes_b[x].avg for x in maes_b.keys()}
                test_time={'test_time@': batch_time_b}
    
            elif start_epoch==config['epochs']:
                batch_time_b, losses_b, maes_b=trainer.test(loader=test_loader, 
                                                            epoch=epoch,
                                                            direct=config['direct'],
                                                            loss_weight_force=weight_f,
                                                            loss_weight_energy=weight_e)            
                test_losses={f'{x}_loss@test':losses_b[x].avg for x in losses_b.keys()}
                test_maes={f'{x}_mae@test':maes_b[x].avg for x in maes_b.keys()}
                test_time={'time@test': batch_time_b}
            else:
                pass

        best_mae_error=min(maes_v['total'].avg, best_mae_error)
        b_mae_error={'best_mae_error':best_mae_error}
        train_losses={f'{x}_loss@train':losses_t[x].avg for x in losses_t.keys()}
        train_maes={f'{x}_mae@train':maes_t[x].avg for x in maes_t.keys()}
        train_time={'time@train': batch_time_t}
        valid_losses={f'{x}_loss@valid':losses_v[x].avg for x in losses_v.keys()}
        valid_maes={f'{x}_mae@valid':maes_v[x].avg for x in maes_v.keys()}
        valid_time={'time@valid': batch_time_v}

        try:
            logged_data={
                        **test_time,
                        **b_mae_error,
                        **train_losses,
                        **train_maes,
                        **train_time,
                        **valid_losses,
                        **valid_maes,
                        **valid_time,
                        **test_losses,
                        **test_maes,
                        **lr_log
                        }
        except:
            logged_data={
                        **b_mae_error,
                        **train_losses,
                        **train_maes,
                        **train_time,
                        **valid_losses,
                        **valid_maes,
                        **valid_time,
                        **lr_log
                        }
        if config['wandb'] :
            wandb.log(logged_data)

        best_path=os.path.join(config['path'], config['save']['best_save'])
        file_path=os.path.join(config['path'], config['save']['check_save'])

        trainer.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizers':{key : normalizers[key].state_dict() for key in normalizers.keys() },
            'scale_datas':scale_datas,
            'args': config,
            'lr_log' : lr_log
        }, is_best,bestname=best_path,filename=file_path)
        
        if (losses_v['total'].avg+losses_t['total'].avg )/2 < config['stop']['early_stop']:
           print('reached required accurcy so stop training')
           break

    if config['wandb'] :
        wandb.finish()

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config={}
    return config

def main():
    parser=argparse.ArgumentParser(description="Run training or other operations")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args=parser.parse_args()
    config=load_config(args.config)
    train(config)
    
