
import os
import copy
import torch
import random
import pickle
import argparse
import warnings
import functools
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pymatgen.core.structure import Structure
from cignn.model.invariant_CNMD import InvCNMD_Q
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class CrystalDataCIF(Dataset):
    def __init__(self, 
                 data_site,
                 data_list, 
                 device,
                 shuffle=False,
                 nbr_mode='nbr', # nbr or distance
                 max_num_nbr=12,
                 radius=5, # distance radius
                 small_angle=True, 
                 angle_cutoff=3.5, # angle radius
                 charge =False,
                 q_model=None,
                 atom_type=[8, 72],
                 interval=1,
                 local_cutoff_type='manual'):

        warnings.filterwarnings('ignore')
        
        self.radius=radius
        self.shuffle=shuffle
        self.nbr_mode=nbr_mode
        self.atom_type=atom_type
        self.data_site=data_site
        self.data_list=data_list
        self.max_num_nbr=max_num_nbr
        
        assert os.path.exists(data_site), 'data_site does not exist!'
        
        self.interval=interval
        if self.interval > 1:
            torch.arange(0,len(self.data_list),self.interval)
        
        file_site=os.path.join(data_site, data_list[0])
        assert os.path.exists(file_site), f'{data_list[0]} does not exist!'

        df_temp=np.load(file_site, allow_pickle=True).tolist()
        self.total_data=df_temp[::self.interval]
            
        self.structures_index=list(range(len(self.total_data)))
        if self.shuffle:
            random.shuffle(self.structures_index)
        
        self.total_data=np.array(self.total_data)[self.structures_index]
        self.total_data=self.total_data.tolist()
        self.device=device

        self.small_angle=small_angle
        self.angle_cutoff=angle_cutoff
        self.local_cutoff_type=local_cutoff_type  # mean or kde
        
        self.charge=charge
        self.qeq_site=q_model
        if self.charge:
            if self.qeq_site != 'None':
                assert os.path.exists(self.qeq_site), 'pre-trained Q model does not exist!'
                self.qeq_model=self.load_qeq(self.qeq_site)
        
    def __len__(self):
        return len(self.total_data)

    def gen_graph(self,coords,small_angle,atom_number,cell):
        device=self.device
        nbr_mode=self.nbr_mode
        max_num_nbr=self.max_num_nbr
        angle_cutoff=self.angle_cutoff

        cart_coords=torch.tensor(coords,device=device)
        total_atoms=len(cart_coords)

        ## get offset data 
        max_offset=1
        total_offset=max_offset*2+1
        coords=torch.arange(-max_offset, max_offset + 1, device=device)
        x, y, z=torch.meshgrid(coords, coords, coords)
        offsets=torch.stack([x, y, z], dim=-1).view(-1, 3).unsqueeze(1)
        new_offsets=torch.zeros(offsets.size(0), 3, 3, device=device,dtype=torch.long)
        rows=torch.arange(3, device=device).unsqueeze(0).expand(offsets.size(0), 3)
        cols=torch.arange(3, device=device).unsqueeze(0).expand(offsets.size(0), 3)
        new_offsets[:, rows, cols]=offsets
        offsets=new_offsets
        offset_points=cart_coords[:, None, :] + torch.sum(torch.matmul(offsets[None, :, :].to(float),cell),dim=-2)

        ## get total point datas involve the offset sites
        total_ori=torch.cat([torch.arange(total_atoms,device=device).reshape(-1,1,1).repeat(1,total_offset**3,1),
                              offset_points,
                              torch.sum(offsets,dim=-2)[None, :, :].repeat(total_atoms, 1, 1)], dim=-1) 
        
        ## imaginary origin offset
        offset_index=torch.concat([torch.arange(0,13),torch.arange(14,27)],axis=-1)
        self_offset=torch.sum(new_offsets,dim=-1)[offset_index]

        ## nbrs data
        re_total=total_ori.view(-1, 7) #(atom_site_number,x,y,z,x_offset,y_offset,z_offset)
        vector=re_total[:, 1:4][None, :] - cart_coords[:, None] #(vector for all atoms)
        
        if nbr_mode == 'nbr':
            ## select nbrs using max_num_nbr
            nearest_indices=torch.topk(torch.norm(vector,dim=-1), max_num_nbr+1, dim=-1, largest=False).indices # nbr+1 because self distance
            nearest_vectors=torch.gather(vector, 1, nearest_indices[:,1:].unsqueeze(2).expand(-1, -1, 3))
            all_nbrs_torch=re_total[nearest_indices[:,1:]].view(-1,7)
            ori_atom_idx=torch.arange(cart_coords.shape[0],device=device).reshape(-1,1,1).repeat(1,max_num_nbr,1).reshape(-1,1)
            nbr_direct=torch.concat([ori_atom_idx,all_nbrs_torch,nearest_vectors.view(-1,3)],dim=-1) #(origin,nbr,x,y,z,x_off,y_off,z_off.vx,vy,vz)
        
        elif nbr_mode == 'distance':
            ##select nbrs using cutoff distance
            _,sorted_indices =torch.sort(torch.norm(vector,dim=-1), dim=-1)
            all_nbrs_torch=re_total[sorted_indices]
            batch_indices=torch.arange(cart_coords.shape[0],device=device).view(cart_coords.shape[0], 1).expand_as(sorted_indices)
            sorted_vector=vector[batch_indices, sorted_indices, :]
            ori_atom_idx=batch_indices.unsqueeze(-1)
            nbr_direct_total=torch.concat([ori_atom_idx,all_nbrs_torch,sorted_vector],dim=-1) #(origin,nbr,x,y,z,x_off,y_off,z_off.vx,vy,vz)
            nbr_direct_total=nbr_direct_total.reshape(-1,11)
            total_distance=torch.norm(nbr_direct_total[:,8:],dim=-1)
            nbr_index=torch.where((total_distance<self.radius) & (total_distance>0.1) )[0]
            nbr_direct=nbr_direct_total[nbr_index] #(origin,nbr,x,y,z,x_off,y_off,z_off.vx,vy,vz)
        
        else:
            print('check nbr mode')

        ## Sort the index columns so that swapped indices become the same
        indices, _=torch.sort(nbr_direct[:, :2], dim=-1)
        vectors=nbr_direct[:, 8:]
        off_sets=nbr_direct[:,5:8]
        data_sorted=torch.cat([indices, vectors,off_sets], dim=-1) # Form the new tensor
        swapped_rows=(data_sorted[:, 0] != nbr_direct[:, 0]) # Find the rows where swapping occurred
        data_sorted[swapped_rows, 2:] *= -1 
        data_sorted=torch.round(data_sorted * 10**5) / 10**5
        nbr_unique, _=torch.unique(data_sorted, dim=0, return_inverse=True) # Use torch.unique with dim=0 to remove duplicates
        nbr_swap=torch.concat([nbr_unique[:, [1, 0]], -1 * nbr_unique[:, 2:]], axis=-1) # b-> a #(nbr,origin,-vx,-vy,-vz)
        total_nbr=torch.concat([nbr_unique, nbr_swap], axis=0) #(origin,nbr,vx,vy,vz,offx,offy,offz)
        N_unique_edges=int(len(nbr_unique)) # Calculate the number of undirected edges in the array
        ina=torch.arange(N_unique_edges)
        nbr_idx_swap=torch.concat([ina + N_unique_edges, ina])

        distance=torch.norm(total_nbr[:,2:5],dim=-1)
        if self.local_cutoff_type == 'mean':
            cut_off=torch.round(distance.mean(),decimals=2)
        elif self.local_cutoff_type == 'nn':
            nn_count=1
            which_nn=nn_count
            all_distance=torch.norm(vector,dim=-1)
            sorted_all_distance=torch.sort(all_distance,dim=-1).values
            target_distance=sorted_all_distance[:,1:]
            round_data=torch.round(target_distance)
            check_data=round_data[:,1:]-round_data[:,:-1]
            x_axis,y_axis=torch.where(check_data>=1)
            _,counts=torch.unique(x_axis,return_counts=True)
            counts=counts.cumsum(dim=-1)
            index=torch.Tensor([which_nn]).repeat(len(counts))
            index=index.to(device)
            new_count=torch.concat([torch.zeros([1],device=device),counts[:-1]],dim=-1)
            new_index=(index+new_count).long()
            new_y=y_axis[new_index].reshape(-1,1)
            nn_distance=torch.gather(target_distance,1,new_y+1)
            nn_distance=nn_distance.mean()
            cut_off=torch.round(nn_distance,decimals=2)
        elif self.local_cutoff_type == 'manual':
            cut_off=angle_cutoff
        else:
            print('check graph cutoff type')

        self.angle_cutoff=cut_off
    
        # select angles, a->b-> c
        angle_fea=torch.zeros(1)
        if small_angle == True:
            while len(angle_fea) < 10 :
                angle_target_idx=torch.where(torch.norm(total_nbr[:,2:5],dim=-1)<self.angle_cutoff)[0]
                angle_target_nbr=total_nbr[angle_target_idx]
                a=angle_target_nbr[:, 0] # center of angle it must in the cell
                b=angle_target_nbr[:, 1] # one neighbor of a atom
                ab_vec=angle_target_nbr[:, 2:5]
                condition_1=(b[:, None] == a) # share same center b-c
                ab_nbr, bc_nbr=torch.where(condition_1)
                condition_2=torch.any(ab_vec[ab_nbr] != -1*(ab_vec[bc_nbr]),axis=-1) # check same nbr 
                bc_nbr=angle_target_idx[bc_nbr[condition_2]]
                ab_nbr=angle_target_idx[ab_nbr[condition_2]]
                angle_fea=torch.concat([total_nbr[:,0:2][ab_nbr], #expand a->b number same with b-c number
                                    total_nbr[:,1][bc_nbr].reshape(-1,1)],  # c
                                    axis=-1) #[a,b,c,a-b vect,b-c vector] atom index
                angle_nbr_idx=torch.stack([ab_nbr,bc_nbr],axis=-1)
                if len(angle_fea) <10:
                    self.angle_cutoff=self.angle_cutoff + 0.1
                # raise ValueError(f'angle_fea is {len(angle_fea)}, structure_idx : {self.structure_idx}')
        else:
            a=total_nbr[:, 0] # center of angle it must in the cell
            b=total_nbr[:, 1] # one neighbor of a atom
            ab_vec=total_nbr[:, 2:5]
            condition_1=(b[:, None] == a) # share same center b-c
            ab_nbr, bc_nbr=torch.where(condition_1)
            condition_2=torch.any(ab_vec[ab_nbr] != -1*(ab_vec[bc_nbr]),axis=-1) # check same nbr 
            bc_nbr=bc_nbr[condition_2]
            ab_nbr=ab_nbr[condition_2]
            angle_fea=torch.concat([total_nbr[:,0:2][ab_nbr], #expand a->b number same with b-c number
                                b[bc_nbr].reshape(-1,1)],  # c
                                axis=-1) #[a,b,c,a-b vect,b-c vector] atom index
            angle_nbr_idx=torch.stack([ab_nbr,bc_nbr],axis=-1)

        #gen datas
        data={}
        data['position']=cart_coords
        data['cell']=cell
        data['atom_number']=atom_number
        data['nbr_fea_idx']=total_nbr[:,:2].to(torch.long)
        data['nbr_swap_idx']=nbr_idx_swap.to(torch.long)
        data['nbr_off_set']=total_nbr[:,5:]
        data['angle_fea_idx']=angle_fea[:,:3].to(torch.long)
        data['angle_nbr_idx']=angle_nbr_idx.to(torch.long)
        data['i_origin_offset']=self_offset
        data['structure_idx']=self.structure_idx

        for key,value in data.items():
            data[key]=value.to('cpu')
        return data

    def load_qeq(self,model_path):
        self.qeq_site=model_path
        model_data=torch.load(self.qeq_site,map_location=lambda storage, loc: storage)
        model_args=argparse.Namespace(**model_data['args'])
        atom_type=torch.tensor(self.atom_type, dtype=torch.long).to(device=self.device)
        model=InvCNMD_Q(atom_type=atom_type,
                        atom_fea_len=model_args.model['atom_fea_len'],
                        nbr_fea_len=model_args.model['nbr_fea_len'],
                        n_conv=model_args.model['n_conv'],
                        num_radial=model_args.model['num_radial'],
                        lmax=model_args.model['lmax'],
                        direct=model_args.model['direct'],
                        cutoff=model_args.model['radius'],
                    )
        model.load_state_dict(model_data['state_dict'])
        model.to(device=self.device)
        model.eval()
        return model
    
    def predict_qeq(self,model,dataset):
        data_temp=copy.deepcopy(dataset)
        n_i=data_temp['position'].shape[0]
        e_i=data_temp['nbr_fea_idx'].shape[0]
        a_i=data_temp['angle_fea_idx'].shape[0]
        data_temp['cell'] =data_temp['cell'].unsqueeze(0)  
        data_temp['crystal_atom_idx']=torch.LongTensor([0]*n_i)
        data_temp['crystal_edge_idx']=torch.LongTensor([0]*e_i)
        data_temp['crystal_angle_idx']=torch.LongTensor([0]*a_i)
        out_data={}
        with torch.no_grad():
            for key, value in data_temp.items():
                data_temp[key]=value.to(self.device)
            out=model(data_temp)
            for target in ['chi','charge','static_e','A_matrix_ii','A_matrix_ij','static_e_atom']:
                out_data[target]=out[target].cpu()
            del data_temp
            del out
            return out_data

    @functools.lru_cache(maxsize=None) # Cache loaded structures
    def __getitem__(self,idx):
        try:
            data=self.total_data[idx]
            force=data['F']
            energy=data['E']
            coords=data['coords']
            atom_number=data['atom_number']
            cell=data['lattice']
            structure_idx=self.structures_index[idx]
            self.structure_idx=torch.LongTensor([int(structure_idx)])
            if self.charge:
                if self.qeq_site == 'None':
                    charge=data['Q']
                    charge=torch.FloatTensor(charge).unsqueeze(1).repeat(1,3)
        except:
            raise ValueError('Data is not valid')

        atom_number=torch.tensor(atom_number, device=self.device)
        cell=torch.tensor(cell, device=self.device) # type: ignore
        data=self.gen_graph(coords=coords, small_angle=self.small_angle, atom_number=atom_number, cell=cell)
        data['force']=torch.FloatTensor(force)
        data['energy']=torch.FloatTensor([energy])
        atom_type,atom_type_count=torch.unique(data['atom_number'], return_counts=True)
        atom_type=atom_type.to(torch.long)
        data['atom_type']=torch.zeros(97)
        data['atom_type'][atom_type]=atom_type_count.to(torch.float)
        if self.charge:
            if self.qeq_site != 'None':
                out_data=self.predict_qeq(self.qeq_model,data)
                data.update(out_data)
            else:
                data['charge']=charge
        for key, value in data.items():
                data[key]=value.to('cpu')
        return data

class StructureData:
    def __init__(self, 
                 structure_data,
                 data_type,
                 device,
                 atom_type=[8,72],
                 nbr_mode='nbr',
                 max_num_nbr=12,
                 radius=8,
                 small_angle=True,
                 angle_cutoff=5,
                 local_cutoff_type='mean',
                 charge =False,
                 e_field=None,
                 q_model=None,
                ):
        warnings.filterwarnings('ignore')
        self.structure_data=structure_data
        if data_type == 'site':
            self.structure=Structure.from_file(self.structure_data)
        elif data_type == 'data':
            self.structure=structure_data
        else:
            ValueError('check data type')

        self.charge=charge
        self.radius=radius
        self.device=device
        self.e_field=e_field
        self.nbr_mode=nbr_mode
        self.atom_type=atom_type
        self.small_angle=small_angle
        self.max_num_nbr=max_num_nbr
        self.angle_cutoff=angle_cutoff
        self.local_cutoff_type=local_cutoff_type
        
        self.qeq_site=q_model
        if self.charge:
            if self.qeq_site != 'None':
                assert os.path.exists(self.qeq_site), 'pre-trained Q model does not exist!'
                self.qeq_model=self.load_qeq(self.qeq_site)
        
    def gen_graph(self,atom_number,cell):
        device=self.device
        nbr_mode=self.nbr_mode
        structure=self.structure
        small_angle=self.small_angle
        max_num_nbr=self.max_num_nbr
        angle_cutoff=self.angle_cutoff

        cart_coords=torch.tensor(structure.cart_coords,device=device)
        total_atoms=len(structure)

        ## get offset data 
        max_offset=1  # adjust according to your requirements
        total_offset=max_offset*2+1
        coords=torch.arange(-max_offset, max_offset + 1, device=device)
        x, y, z=torch.meshgrid(coords, coords, coords)
        offsets=torch.stack([x, y, z], dim=-1).view(-1, 3).unsqueeze(1)
        new_offsets=torch.zeros(offsets.size(0), 3, 3, device=device,dtype=torch.long)
        rows=torch.arange(3, device=device).unsqueeze(0).expand(offsets.size(0), 3)
        cols=torch.arange(3, device=device).unsqueeze(0).expand(offsets.size(0), 3)
        new_offsets[:, rows, cols]=offsets
        offsets=new_offsets
        offset_points=cart_coords[:, None, :] + torch.sum(torch.matmul(offsets[None, :, :].to(float),cell),dim=-2)

        ## get total point datas involve the offset sites
        total_ori=torch.cat([torch.arange(total_atoms,device=device).reshape(-1,1,1).repeat(1,total_offset**3,1),
                             offset_points,
                             torch.sum(offsets,dim=-2)[None, :, :].repeat(total_atoms, 1, 1)], dim=-1) 
        
        ## imaginary origin offset
        offset_index=torch.concat([torch.arange(0,13),torch.arange(14,27)],axis=-1)
        self_offset=torch.sum(new_offsets,dim=-1)[offset_index]

        ## nbrs data
        re_total=total_ori.view(-1, 7) #(atom_site_number,x,y,z,x_offset,y_offset,z_offset)
        vector=re_total[:, 1:4][None, :] - cart_coords[:, None] #(vector for all atoms)
        if nbr_mode == 'nbr':
            nearest_indices=torch.topk(torch.norm(vector,dim=-1), max_num_nbr+1, dim=-1, largest=False).indices # nbr+1 because self distance
            nearest_vectors=torch.gather(vector, 1, nearest_indices[:,1:].unsqueeze(2).expand(-1, -1, 3))
            all_nbrs_torch=re_total[nearest_indices[:,1:]].view(-1,7)
            ori_atom_idx=torch.arange(cart_coords.shape[0],device=device).reshape(-1,1,1).repeat(1,max_num_nbr,1).reshape(-1,1)
            nbr_direct=torch.concat([ori_atom_idx,all_nbrs_torch,nearest_vectors.view(-1,3)],dim=-1) #(origin,nbr,x,y,z,x_off,y_off,z_off.vx,vy,vz)
        elif nbr_mode == 'distance':
            _,sorted_indices =torch.sort(torch.norm(vector,dim=-1), dim=-1)
            all_nbrs_torch=re_total[sorted_indices]
            batch_indices=torch.arange(cart_coords.shape[0],device=device).view(cart_coords.shape[0], 1).expand_as(sorted_indices)
            sorted_vector=vector[batch_indices, sorted_indices, :]
            ori_atom_idx=batch_indices.unsqueeze(-1)
            nbr_direct_total=torch.concat([ori_atom_idx,all_nbrs_torch,sorted_vector],dim=-1) #(origin,nbr,x,y,z,x_off,y_off,z_off.vx,vy,vz)
            nbr_direct_total=nbr_direct_total.reshape(-1,11)
            total_distance=torch.norm(nbr_direct_total[:,8:],dim=-1)
            nbr_index=torch.where((total_distance<self.radius) & (total_distance>0.1) )[0]
            nbr_direct=nbr_direct_total[nbr_index] #(origin,nbr,x,y,z,x_off,y_off,z_off.vx,vy,vz)
        else:
            print('check nbr mode')

        ## Sort the index columns so that swapped indices become the same
        indices, _=torch.sort(nbr_direct[:, :2], dim=-1)
        vectors=nbr_direct[:, 8:]
        off_sets=nbr_direct[:,5:8]
        data_sorted=torch.cat([indices, vectors,off_sets], dim=-1) # Form the new tensor
        swapped_rows=(data_sorted[:, 0] != nbr_direct[:, 0]) # Find the rows where swapping occurred
        data_sorted[swapped_rows, 2:] *= -1 
        data_sorted=torch.round(data_sorted * 10**5) / 10**5
        nbr_unique, _=torch.unique(data_sorted, dim=0, return_inverse=True) # Use torch.unique with dim=0 to remove duplicates
        nbr_swap=torch.concat([nbr_unique[:, [1, 0]], -1 * nbr_unique[:, 2:]], axis=-1) # b-> a #(nbr,origin,-vx,-vy,-vz)
        total_nbr=torch.concat([nbr_unique, nbr_swap], axis=0) #(origin,nbr,vx,vy,vz,offx,offy,offz)
        N_unique_edges=int(len(nbr_unique)) # Calculate the number of undirected edges in the array
        ina=torch.arange(N_unique_edges)
        nbr_idx_swap=torch.concat([ina + N_unique_edges, ina])

        distance=torch.norm(total_nbr[:,2:5],dim=-1)
        if self.local_cutoff_type == 'mean':
            cut_off=torch.round(distance.mean(),decimals=2)
        elif self.local_cutoff_type == 'nn':
            nn_count=1
            which_nn=nn_count
            all_distance=torch.norm(vector,dim=-1)
            sorted_all_distance=torch.sort(all_distance,dim=-1).values
            target_distance=sorted_all_distance[:,1:]
            round_data=torch.round(target_distance)
            check_data=round_data[:,1:]-round_data[:,:-1]
            x_axis,y_axis=torch.where(check_data>=1)
            _,counts=torch.unique(x_axis,return_counts=True)
            counts=counts.cumsum(dim=-1)
            index=torch.Tensor([which_nn]).repeat(len(counts))
            index=index.to(device)
            new_count=torch.concat([torch.zeros([1],device=device),counts[:-1]],dim=-1)
            new_index=(index+new_count).long()
            new_y=y_axis[new_index].reshape(-1,1)
            nn_distance=torch.gather(target_distance,1,new_y+1)
            nn_distance=nn_distance.mean()
            cut_off=torch.round(nn_distance,decimals=2)
        elif self.local_cutoff_type == 'manual':
            cut_off=angle_cutoff
        else:
            print('check graph cutoff type')

        self.angle_cutoff=cut_off
        # select angles, a->b-> c
        angle_fea=torch.zeros(1)
        if small_angle == True:
            while len(angle_fea) < 10 :
                angle_target_idx=torch.where(torch.norm(total_nbr[:,2:5],dim=-1)<self.angle_cutoff)[0]
                angle_target_nbr=total_nbr[angle_target_idx]
                a=angle_target_nbr[:, 0] # center of angle it must in the cell
                b=angle_target_nbr[:, 1] # one neighbor of a atom
                ab_vec=angle_target_nbr[:, 2:5]
                condition_1=(b[:, None] == a) # share same center b-c
                ab_nbr, bc_nbr=torch.where(condition_1)
                condition_2=torch.any(ab_vec[ab_nbr] != -1*(ab_vec[bc_nbr]),axis=-1) # check same nbr 
                bc_nbr=angle_target_idx[bc_nbr[condition_2]]
                ab_nbr=angle_target_idx[ab_nbr[condition_2]]
                angle_fea=torch.concat([total_nbr[:,0:2][ab_nbr], #expand a->b number same with b-c number
                                    total_nbr[:,1][bc_nbr].reshape(-1,1)],  # c
                                    axis=-1) #[a,b,c,a-b vect,b-c vector] atom index
                angle_nbr_idx=torch.stack([ab_nbr,bc_nbr],axis=-1)
                if len(angle_fea) <10:
                    self.angle_cutoff=self.angle_cutoff + 0.1
        else:
            a=total_nbr[:, 0] # center of angle it must in the cell
            b=total_nbr[:, 1] # one neighbor of a atom
            ab_vec=total_nbr[:, 2:5]
            condition_1=(b[:, None] == a) # share same center b-c
            ab_nbr, bc_nbr=torch.where(condition_1)
            condition_2=torch.any(ab_vec[ab_nbr] != -1*(ab_vec[bc_nbr]),axis=-1) # check same nbr 
            bc_nbr=bc_nbr[condition_2]
            ab_nbr=ab_nbr[condition_2]
            angle_fea=torch.concat([total_nbr[:,0:2][ab_nbr], #expand a->b number same with b-c number
                                b[bc_nbr].reshape(-1,1)],  # c
                                axis=-1) #[a,b,c,a-b vect,b-c vector] atom index
            angle_nbr_idx=torch.stack([ab_nbr,bc_nbr],axis=-1)
        
        #gen datas
        data={}
        data['position']=cart_coords
        data['cell']=cell
        data['atom_number']=atom_number
        data['nbr_fea_idx']=total_nbr[:,:2].to(torch.long)
        data['nbr_swap_idx']=nbr_idx_swap.to(torch.long)
        data['nbr_off_set']=total_nbr[:,5:]
        data['angle_fea_idx']=angle_fea[:,:3].to(torch.long)
        data['angle_nbr_idx']=angle_nbr_idx.to(torch.long)
        data['i_origin_offset']=self_offset

        n_i=data['position'].shape[0]
        e_i=data['nbr_fea_idx'].shape[0]
        a_i=data['angle_fea_idx'].shape[0]
        data['cell'] =data['cell'].unsqueeze(0)  
        data['crystal_atom_idx']=torch.LongTensor([0]*n_i)
        data['crystal_edge_idx']=torch.LongTensor([0]*e_i)
        data['crystal_angle_idx']=torch.LongTensor([0]*a_i)

        for key,value in data.items():
            data[key]=value.to('cpu')
        return data

    def load_qeq(self,model_path):
        self.qeq_site=model_path
        model_data=torch.load(self.qeq_site,map_location=lambda storage, loc: storage)
        model_args=argparse.Namespace(**model_data['args'])
        atom_type=torch.Tensor(list(self.atom_type)).reshape(-1,1).to(device=self.device)
        model=InvCNMD_Q(atom_type=atom_type,
                        atom_fea_len=model_args.model['atom_fea_len'],
                        nbr_fea_len=model_args.model['nbr_fea_len'],
                        n_conv=model_args.model['n_conv'],
                        num_radial=model_args.model['num_radial'],
                        lmax=model_args.model['lmax'],
                        direct=model_args.model['direct'],
                        cutoff=model_args.model['radius'],
                        e_field=self.e_field,
                        )
        model.load_state_dict(model_data['state_dict'])
        model.to(device=self.device)
        model.eval()
        return model
    
    def predict_qeq(self,model,dataset):
        data_temp=copy.deepcopy(dataset)
        out_data={}
        with torch.no_grad():
            for key, value in data_temp.items():
                data_temp[key]=value.to(self.device)
            out=model(data_temp)
            for target in ['chi','charge','static_e','A_matrix_ii','A_matrix_ij','static_e_atom']:
                out_data[target]=out[target].cpu()
            try:
                for target in ['field_chi', 'field_charge']:
                    out_data[target]=out[target].cpu()
            except:
                pass
            del data_temp
            del out
            return out_data
        
    def get_data(self,device):
        atom_number=torch.tensor(list(map(lambda x: x.number,self.structure.species)), device=self.device)
        cell=torch.tensor(self.structure.lattice.matrix, device=self.device) # type: ignore
        data= self.gen_graph(atom_number=atom_number, cell=cell)
        atom_type,atom_type_count=torch.unique(data['atom_number'], return_counts=True)
        atom_type=atom_type.to(torch.long)
        data['atom_type']=torch.zeros(97)
        data['atom_type'][atom_type]=atom_type_count.to(torch.float)
        if self.charge:
            if self.qeq_site == 'None':
                raise ValueError('check qeq site')
            else:
                out_data=self.predict_qeq(self.qeq_model,data)
                data.update(out_data)
        data=dict(zip(data.keys(),map(lambda x : x.to(device),data.values())))
        return data

def one_data(data:dict,device):
    n_i=data['position'].shape[0]
    e_i=data['nbr_fea_idx'].shape[0]
    a_i=data['angle_fea_idx'].shape[0]
    data['cell']=data['cell'].unsqueeze(0)  
    data['crystal_atom_idx']=torch.LongTensor([0]*n_i)
    data['crystal_edge_idx']=torch.LongTensor([0]*e_i)
    data['crystal_angle_idx']=torch.LongTensor([0]*a_i)
    data=dict(zip(data.keys(),map(lambda x : x.to(device),data.values())))
    return data

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
        eps=1e-8
        self.rms = torch.sqrt(torch.mean(tensor**2)+eps)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def scale(self,tensor):
        return (tensor)/self.rms

    def descale(self,scaled_tensor):
        return scaled_tensor * self.rms

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std,
                'rms': self.rms}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.rms = state_dict['rms']

def collate_pool(dataset_list):
    merged_data={}
    base_atom_idx=0
    base_nbr_idx=0
    base_angle_idx=0
    base_graph_type_idx=0
    for i,d in enumerate(dataset_list):
        temp_data=copy.deepcopy(d)
       
        n_i=temp_data['position'].shape[0]
        e_i=temp_data['nbr_fea_idx'].shape[0]
        a_i=temp_data['angle_fea_idx'].shape[0]  

        if any(word.startswith('intra_') for word in d.keys()):
            g_i=d['intra_graph_idx'].shape[0]
            ge_i=d['inter_graph_idx'].shape[0]
            temp_data['graph_type']=d['graph_type']+base_graph_type_idx
            temp_data['poly_idx']=d['poly_idx']+base_atom_idx
            temp_data['intra_graph_idx']=d['intra_graph_idx'] + base_nbr_idx
            temp_data['inter_graph_idx']=d['inter_graph_idx'] + base_nbr_idx
            temp_data['crystal_intra_graph_idx']=torch.LongTensor([i]* g_i)
            temp_data['crystal_inter_graph_idx']=torch.LongTensor([i] * ge_i)
            base_graph_type_idx += 2     

        temp_data['nbr_fea_idx']=d['nbr_fea_idx'] + base_atom_idx
        temp_data['angle_fea_idx']=d['angle_fea_idx'] + base_atom_idx

        temp_data['nbr_swap_idx']=d['nbr_swap_idx'] + base_nbr_idx
        temp_data['angle_nbr_idx']=d['angle_nbr_idx'] + base_nbr_idx

        temp_data['crystal_atom_idx']=torch.LongTensor([i]*n_i)
        temp_data['crystal_edge_idx']=torch.LongTensor([i]*e_i)
        temp_data['crystal_angle_idx']=torch.LongTensor([i]*a_i)

        for key, value in temp_data.items():
            if key in merged_data:
                merged_data[key].append(value)
            else:
                merged_data[key]=[value]
        base_atom_idx=base_atom_idx + n_i
        base_nbr_idx=base_nbr_idx + e_i
        base_angle_idx=base_angle_idx + a_i

    merged_data['energy']=torch.stack(merged_data['energy']) #batch,1
    merged_data['force']=torch.concat(merged_data['force'])  #atoms ,3
    merged_data['position']=torch.concat(merged_data['position']) #atoms ,3
    merged_data['cell']=torch.stack(merged_data['cell']) #batch 3,3,
    merged_data['crystal_atom_idx']=torch.concat(merged_data['crystal_atom_idx']) #atoms,
    merged_data['crystal_edge_idx']=torch.concat(merged_data['crystal_edge_idx']) #edgs,
    merged_data['crystal_angle_idx']=torch.concat(merged_data['crystal_angle_idx']) #angles,
    merged_data['atom_number']=torch.concat(merged_data['atom_number']) #atoms,
    merged_data['nbr_fea_idx']=torch.concat(merged_data['nbr_fea_idx']) #edgs,2
    merged_data['nbr_swap_idx']=torch.concat(merged_data['nbr_swap_idx']) #edgs,
    merged_data['nbr_off_set']=torch.concat(merged_data['nbr_off_set']) #edgs,3
    merged_data['angle_fea_idx']=torch.concat(merged_data['angle_fea_idx']) #angle,3
    merged_data['angle_nbr_idx']=torch.concat(merged_data['angle_nbr_idx']) #angle,2
    merged_data['i_origin_offset']=torch.stack(merged_data['i_origin_offset']) #batch, 26, ,3
    merged_data['structure_idx']=torch.concat(merged_data['structure_idx'])
    merged_data['atom_type']=torch.stack(merged_data['atom_type'])
    try:
        merged_data['charge']=torch.concat(merged_data['charge'])
        merged_data['A_matrix_ii']=torch.concat(merged_data['A_matrix_ii']) 
        merged_data['A_matrix_ij']=torch.concat(merged_data['A_matrix_ij'])
        merged_data['chi']=torch.concat(merged_data['chi'])
        merged_data['static_e']=torch.concat(merged_data['static_e'])
        merged_data['static_e_atom']=torch.concat(merged_data['static_e_atom'])
    except:
        pass

    return merged_data

def get_loader(train_dataset, val_dataset, test_dataset=False, collate_fn=default_collate, batch_size=64, return_test=False, num_workers=1, pin_memory=False, **kwargs):
    train_indices=list(range(len(train_dataset)))
    train_sampler=SubsetRandomSampler(train_indices)
    val_indices=list(range(len(val_dataset)))
    val_sampler=SubsetRandomSampler(val_indices)
    train_loader=DataLoader(train_dataset, 
                            batch_size=batch_size,
                            sampler=train_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, 
                            pin_memory=pin_memory)
    val_loader=DataLoader(val_dataset, 
                          batch_size=batch_size,
                          sampler=val_sampler,
                          num_workers=num_workers,
                          collate_fn=collate_fn, 
                          pin_memory=pin_memory)                            
    if return_test and test_dataset:
        test_loader=DataLoader(test_dataset, 
                               batch_size=batch_size,
                               sampler=test_sampler,
                               num_workers=num_workers,
                               collate_fn=collate_fn, 
                               pin_memory=pin_memory)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, val_loader

def cumsum_from_zero(input_, device):
    cumsum=torch.zeros_like(input_, device=device)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum

def split_data(data_config, test=True):
    data_site=os.path.join(data_config['site'], data_config['preprocess']['name'][0])
    df_temp=np.load(data_site, allow_pickle=True).tolist()
    total_data_list=df_temp[::self.interval]
    if data_config['preprocess'].get('train_size') and data_config['preprocess'].get('valid_size'):
        train_size=data_config['preprocess']['train_size']
        valid_size=data_config['preprocess']['valid_size']
        if test:
            test_size=data_config['preprocess']['test_size']
        
        train_data=total_data_list[:train_size]
        valid_data=total_data_list[train_size:train_size + valid_size]
        test_data=total_data_list[train_size + valid_size:] if test else []
    elif data_config['preprocess'].get('train_ratio') and data_config['preprocess'].get('valid_ratio'):
        train_ratio=data_config['preprocess']['train_ratio']
        valid_ratio=data_config['preprocess']['valid_ratio']
        test_ratio=data_config['preprocess']['test_ratio'] if test else 0
        
        total_size=len(total_data_list)
        train_size=int(total_size * train_ratio)
        valid_size=int(total_size * valid_ratio)
        test_size=total_size - train_size - valid_size
        
        train_data=total_data_list[:train_size]
        valid_data=total_data_list[train_size:train_size + valid_size]
        test_data=total_data_list[train_size + valid_size:] if test else []
    else:
        raise ValueError("Invalid preprocess configuration. Provide either size or ratio.")

    # 데이터 저장
    train_site = os.path.join(data_config['site'], 'split_train.npy')
    valid_site = os.path.join(data_config['site'], 'split_valid.npy')
    np.save(train_site, train_data)
    np.save(valid_site, valid_data)
    
    if test:
        test_site = os.path.join(data_config['site'], 'split_test.npy')
        np.save(test_site, test_data)
    
    print(f"Train data saved at: {train_site}")
    print(f"Validation data saved at: {valid_site}")
    if test:
        print(f"Test data saved at: {test_site}")

def save_data(file_path, data):
    """Utility function to save data."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    """Utility function to load data."""
    print('Load:', file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)