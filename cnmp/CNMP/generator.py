import os
import ase
import copy
import torch
import warnings
import numpy as np
import argparse
from cignn.model.invariant_CNMD import InvCNMD_Q

torch.set_default_dtype(torch.float64)

class AtomsData:
    def __init__(self,
                 device,
                 nbr_mode='nbr',
                 max_num_nbr=12,
                 radius=8,
                 small_angle=True,
                 angle_cutoff=5,
                 local_cutoff_type='mean',
                 charge=False,
                 e_field=None,
                 q_model=None,
                 atom_type=[8, 72]):
        warnings.filterwarnings('ignore')
        print('e_field', e_field)
        self.device = device
        self.radius = radius
        self.nbr_mode = nbr_mode
        self.atom_type = atom_type
        self.max_num_nbr = max_num_nbr
        self.small_angle = small_angle
        self.angle_cutoff = angle_cutoff
        self.local_cutoff_type = local_cutoff_type  # or kde
        self.e_field = e_field
        self.charge = charge
        self.qeq_site = q_model
        assert os.path.exists(self.qeq_site), 'pre-trained Q model does not exist!'
        self.qeq_model = self.load_qeq(self.qeq_site)

    def gen_graph(self, atoms: ase.Atoms, sid=None):
        atom_number = torch.tensor(atoms.get_atomic_numbers(), device=self.device)
        cart_coords = torch.tensor(atoms.get_positions(), device=self.device)
        cell = torch.tensor(np.array(atoms.get_cell()), device=self.device).view(3, 3)
        total_atoms = cart_coords.shape[0]
        
        max_offset=1
        total_offset=max_offset*2+1
        coords=torch.arange(-max_offset, max_offset + 1, device=self.device)
        x, y, z=torch.meshgrid(coords, coords, coords)
        offsets=torch.stack([x, y, z], dim=-1).view(-1, 3).unsqueeze(1)
        new_offsets=torch.zeros(offsets.size(0), 3, 3, device=self.device, dtype=torch.long)
        rows=torch.arange(3, device=self.device).unsqueeze(0).expand(offsets.size(0), 3)
        cols=torch.arange(3, device=self.device).unsqueeze(0).expand(offsets.size(0), 3)
        new_offsets[:, rows, cols]=offsets
        offsets=new_offsets
        offset_points=cart_coords[:, None, :] + torch.sum(torch.matmul(offsets[None, :, :].to(float),cell),dim=-2)

        ## get total point datas involve the offset sites
        total_ori=torch.cat([torch.arange(total_atoms,device=self.device).reshape(-1,1,1).repeat(1,total_offset**3,1),
                              offset_points,
                              torch.sum(offsets,dim=-2)[None, :, :].repeat(total_atoms, 1, 1)], dim=-1) 
        
        ## imaginary origin offset
        offset_index=torch.concat([torch.arange(0,13),torch.arange(14,27)],axis=-1)
        self_offset=torch.sum(new_offsets,dim=-1)[offset_index]

        ## nbrs data
        re_total=total_ori.view(-1, 7) #(atom_site_number,x,y,z,x_offset,y_offset,z_offset)
        vector=re_total[:, 1:4][None, :] - cart_coords[:, None] #(vector for all atoms)
        
        if self.nbr_mode == 'nbr':
            nearest_indices=torch.topk(torch.norm(vector,dim=-1), self.max_num_nbr+1, dim=-1, largest=False).indices # nbr+1 because self distance
            nearest_vectors=torch.gather(vector, 1, nearest_indices[:,1:].unsqueeze(2).expand(-1, -1, 3))
            all_nbrs_torch=re_total[nearest_indices[:,1:]].view(-1,7)
            ori_atom_idx=torch.arange(cart_coords.shape[0],device=self.device).reshape(-1,1,1).repeat(1,self.max_num_nbr,1).reshape(-1,1)
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
        if self.small_angle == True:
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

        # generate data
        data = {
            'position': cart_coords,
            'cell': cell,
            'atom_number': atom_number,
            'nbr_fea_idx': total_nbr[:,:2].to(torch.long),
            'nbr_swap_idx': nbr_idx_swap.to(torch.long),
            'nbr_off_set': total_nbr[:,5:],
            'angle_fea_idx': angle_fea[:,:3].to(torch.long),
            'angle_nbr_idx': angle_nbr_idx.to(torch.long),
            'i_origin_offset': self_offset
        }

        n_i = data['position'].shape[0]
        e_i = data['nbr_fea_idx'].shape[0]
        a_i = data['angle_fea_idx'].shape[0]
        data['cell'] = data['cell'].unsqueeze(0)
        data['crystal_atom_idx'] = torch.LongTensor([0] * n_i)
        data['crystal_edge_idx'] = torch.LongTensor([0] * e_i)
        data['crystal_angle_idx'] = torch.LongTensor([0] * a_i)

        for key, value in data.items():
            data[key] = value.to('cpu')
        return data

    def load_qeq(self, model_path):
        self.qeq_site = model_path
        model_data = torch.load(self.qeq_site, map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_data['args'])

        atom_type = torch.tensor(self.atom_type, dtype=torch.long)
        model=InvCNMD_Q(atom_type=atom_type,
                        atom_fea_len=model_args.model['atom_fea_len'],
                        nbr_fea_len=model_args.model['nbr_fea_len'],
                        n_conv=model_args.model['n_conv'],
                        num_radial=model_args.model['num_radial'],
                        lmax=model_args.model['lmax'],
                        direct=model_args.direct,
                        cutoff=model_args.data['radius'],
                        e_field=self.e_field,
                        )
        model.load_state_dict(model_data['state_dict'])
        model.to(device=self.device)
        model.eval()
        return model
    
    def predict_qeq(self, model, dataset):
        data_temp = copy.deepcopy(dataset)
        out_data = {}
        with torch.no_grad():
            for key, value in data_temp.items():
                data_temp[key] = value.to(self.device)
            out = model(data_temp)
            for target in ['chi', 'charge', 'static_e', 'A_matrix_ii', 'A_matrix_ij', 'static_e_atom']:
                out_data[target]=out[target].cpu()
            try:
                for target in ['field_chi', 'field_charge']:
                    out_data[target]=out[target].cpu()
            except:
                pass
            del data_temp
            del out
            return out_data
        
    def get_data(self, atoms: ase.Atoms, sid=None):
        data = self.gen_graph(atoms)
        atom_type, atom_type_count = torch.unique(data['atom_number'], return_counts=True)
        atom_type = atom_type.to(torch.long)
        data['atom_type'] = torch.zeros(97)
        data['atom_type'][atom_type] = atom_type_count.to(torch.float64)
        if self.charge:
            if self.qeq_site == 'None':
                raise ValueError('check qeq site')
            else:
                out_data = self.predict_qeq(self.qeq_model, data)
                data.update(out_data)
        data = dict(zip(data.keys(),map(lambda x : x.to(self.device),data.values())))
        return data