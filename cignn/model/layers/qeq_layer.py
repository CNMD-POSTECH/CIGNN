import math
import torch.nn as nn
from copy import deepcopy
from scipy import constants
from torch_scatter import scatter

from .basic_layers import *
from .basis_layers import *

class QEQ(nn.Module):
    def __init__ (self, atom_fea_len, nbr_fea_len, activation, atom_type, e_field=False, energy_scale=1.0, transfer=False):
        super(QEQ,self).__init__()
        self.e_field = e_field 
        self.atom_fea_len=atom_fea_len
        self.activation=activation
        self.nbr_fea_len=nbr_fea_len
        self.energy_scale=energy_scale

        atom_type = atom_type.squeeze().detach().cpu()
        self.atom_type_number=torch.tensor(atom_type, dtype=torch.long)
        self.to_chi = ResidualLayers(n_layers=4,
                                     n_in=self.atom_fea_len,
                                     n_mid=int(self.atom_fea_len/2),
                                     activation=self.activation)

        self.chi_out = Dense(in_features=self.atom_fea_len,
                        out_features=1,activation=None,bias=False)

        self.to_hardness = ResidualLayers(n_layers=4,
                                          n_in=self.atom_fea_len,
                                          n_mid=int(self.atom_fea_len/2),
                                          activation=self.activation)

        self.hardness_out = Dense(in_features=self.atom_fea_len,
                out_features=1,activation=None,bias=False)

        self.to_A_matrix_ij = ResidualLayers(n_layers=4,
                                             n_in=2*self.atom_fea_len+self.nbr_fea_len,
                                             n_mid=int(self.atom_fea_len/2),
                                             activation=self.activation)

        self.to_A_matrix_ij_out = Dense(in_features=2*self.atom_fea_len+self.nbr_fea_len,
                                        out_features=1,
                                        activation=None,
                                        bias=False)

        self.use_A_matrix=True
        COULOMB_FACTOR = 14.399645478425668 # V*angstrom/e
        self.scaled_coulomb_factor=COULOMB_FACTOR/self.energy_scale
        self.norm=Crystal_Norm(self.atom_fea_len)
        electronegativities = [
                                0,  # 0 placeholder
                                2.20,  # Hydrogen
                                0,  # Helium
                                0.98,  # Lithium
                                1.57,  # Beryllium
                                2.04,  # Boron
                                2.55,  # Carbon
                                3.04,  # Nitrogen
                                3.44,  # Oxygen
                                3.98,  # Fluorine
                                0,  # Neon
                                0.93,  # Sodium
                                1.31,  # Magnesium
                                1.61,  # Aluminium
                                1.90,  # Silicon
                                2.19,  # Phosphorus
                                2.58,  # Sulfur
                                3.16,  # Chlorine
                                0,  # Argon
                                0.82,  # Potassium
                                1.00,  # Calcium
                                1.36,  # Scandium
                                1.61,  # Titanium
                                1.90,  # Vanadium
                                2.19,  # Chromium
                                1.86,  # Manganese
                                1.83,  # Iron
                                1.88,  # Cobalt
                                1.91,  # Nickel
                                1.90,  # Copper
                                1.65,  # Zinc
                                1.81,  # Gallium
                                2.01,  # Germanium
                                2.18,  # Arsenic
                                2.55,  # Selenium
                                2.96,  # Bromine
                                0,  # Krypton
                                0.82,  # Rubidium
                                0.95,  # Strontium
                                1.22,  # Yttrium
                                1.33,  # Zirconium
                                1.60,  # Niobium
                                2.16,  # Molybdenum
                                1.90,  # Technetium
                                2.20,  # Ruthenium
                                2.28,  # Rhodium
                                2.20,  # Palladium
                                1.93,  # Silver
                                1.69,  # Cadmium
                                1.78,  # Indium
                                1.96,  # Tin
                                2.05,  # Antimony
                                2.10,  # Tellurium
                                2.66,  # Iodine
                                0,  # Xenon
                                0.79,  # Caesium
                                0.89,  # Barium
                                1.10,  # Lanthanum
                                1.12,  # Cerium
                                1.13,  # Praseodymium
                                1.14,  # Neodymium
                                0,  # Promethium
                                1.17,  # Samarium
                                0,  # Europium
                                1.20,  # Gadolinium
                                1.22,  # Terbium
                                1.23,  # Dysprosium
                                0,  # Holmium
                                0,  # Erbium
                                0,  # Thulium
                                0,  # Ytterbium
                                0,  # Lutetium
                                1.27,  # Hafnium
                                1.30,  # Tantalum
                                1.50,  # Tungsten
                                2.36,  # Rhenium
                                1.90,  # Osmium
                                2.20,  # Iridium
                                2.20,  # Platinum
                                2.28,  # Gold
                                2.54,  # Mercury
                                2.00,  # Thallium
                                1.62,  # Lead
                                2.33,  # Bismuth
                                2.02,  # Polonium
                                2.0,  # Astatine
                                0,  # Radon
                                0.79,  # Francium
                                0.89,  # Radium
                                1.10,  # Actinium
                                1.30,  # Thorium
                                1.50,  # Protactinium
                                1.38,  # Uranium
                                1.36,  # Neptunium
                                1.28  # Plutonium
                            ]

        self.electronegativities = torch.tensor(electronegativities)

        self.chi_shift = torch.nn.Parameter(self.electronegativities[self.atom_type_number])
        self.chi_shift.requires_grad_()
        self.scale_ii = torch.nn.Parameter(torch.ones(len(self.atom_type_number),requires_grad=True))
        self.shift_ii = torch.nn.Parameter(torch.ones(len(self.atom_type_number),requires_grad=True))

        self.scale_ij = torch.nn.Parameter(torch.zeros(1,requires_grad=True))
        self.shift_ij = torch.nn.Parameter(torch.zeros(1,requires_grad=True))
        self.transfer = transfer
        
    def forward(self,data):
        nbr_fea_idx = data['nbr_fea_idx']
        atom_fea=data['atom_fea']
        nbr_fea=data['edge']
        
        edge_src=nbr_fea_idx[:,0]
        edge_dst=nbr_fea_idx[:,1]
        position=data['position']

        # chi and hardness come from node feature after layer 
        total_nbr=torch.concat([atom_fea[nbr_fea_idx].reshape(-1,self.atom_fea_len*2),nbr_fea],dim=-1)

        if self.e_field:
            e_field_s, e_field_e, axis = self.e_field
            if e_field_s == e_field_e:
                e_field = e_field_s
                center_positions = position[:, axis]
                learned_chi_shift = self.chi_shift.clone()
                chi_shifted_adjustment = learned_chi_shift[data['atom_number']] - (center_positions * (e_field))
                field_chi_shifted_adjustment = chi_shifted_adjustment.to(torch.float32)
            else:
                print('Not implemented')
        
        system_chi_shifted_adjustment = self.chi_shift[data['atom_number']]  # Use the trained nn.Parameter
        
        chi=self.to_chi(atom_fea)
        chi=self.chi_out(chi).squeeze()    
        system_chi = chi.clone()+system_chi_shifted_adjustment
        
        if self.e_field:
            field_chi=chi.clone()+field_chi_shifted_adjustment

        hardness=self.to_hardness(atom_fea)
        hardness=torch.square(self.hardness_out(hardness)).squeeze() * self.scale_ii[data['atom_number']] + self.shift_ii[data['atom_number']]

        # A_matrix_ij is come from edge feature and distance after layer
        total_nbr=torch.concat([atom_fea[nbr_fea_idx].reshape(-1,self.atom_fea_len*2),nbr_fea],dim=-1)
        A_matrix_ij = self.to_A_matrix_ij(total_nbr) # (edge_number,atom_feature)

        #Todo:this layer is right? constrain is in scale_ij shift_ij
        A_matrix_ij = self.to_A_matrix_ij_out(A_matrix_ij).squeeze() * self.scale_ij + self.shift_ij
        A_matrix_ij = A_matrix_ij.clamp_min(1e-7).reshape(-1,1) * (self.scaled_coulomb_factor/data['distance']).reshape(-1,1)
        A_matrix_ii = hardness.reshape(-1,1)
        
        system_chi = self.batch_cal_long(system_chi)
        if self.e_field:
            field_chi = self.batch_cal_long(field_chi)

        system_args = self.prepare_batch_linear_solver_args(data, -system_chi, A_matrix_ij, A_matrix_ii)
        if self.e_field:
            field_args = self.prepare_batch_linear_solver_args(data, -field_chi, A_matrix_ij, A_matrix_ii)
        charge = torch.concat(list(map(self.batch_linear_solver, system_args)),axis=0)
        if self.e_field:
            charge_field = torch.concat(list(map(self.batch_linear_solver, field_args)),axis=0)

        data['A_matrix_ij']=A_matrix_ij 
        data['A_matrix_ii']=A_matrix_ii
        
        if self.e_field:
            data['charge'] = charge.squeeze()
            data['field_charge'] = charge_field.squeeze()
        else:
            data['charge'] = charge.squeeze()

        if self.e_field:
            data['chi'] = system_chi
            data['field_chi'] = field_chi
        else:
            data['chi'] = system_chi
        
        if self.transfer :
            data['atom_charge_fea'] = data['atom_fea']
            data['edge_charge_fea'] = data['edge']
                      
        if self.use_A_matrix:
            e_qeq_ij_x= A_matrix_ij*charge[:,0][edge_src].reshape(-1,1)*charge[:,0][edge_dst].reshape(-1,1)
            e_qeq_ij_y= A_matrix_ij*charge[:,1][edge_src].reshape(-1,1)*charge[:,1][edge_dst].reshape(-1,1)
            e_qeq_ij_z= A_matrix_ij*charge[:,2][edge_src].reshape(-1,1)*charge[:,2][edge_dst].reshape(-1,1)

            e_qeq_ij = torch.mean(torch.cat([e_qeq_ij_x,e_qeq_ij_y,e_qeq_ij_z],dim=-1),dim=-1).reshape(-1,1)
        
            e_qeq_i = 0.5*scatter(e_qeq_ij,edge_src,dim=0,reduce='sum')
            
            e_qeq_ij = scatter(e_qeq_ij,data['crystal_edge_idx'],dim=0,reduce='sum')
            e_qeq_ij = 0.5*e_qeq_ij
            
            e_qeq_ii_x= A_matrix_ii*charge[:,0].reshape(-1,1)*charge[:,0].reshape(-1,1)
            e_qeq_ii_y= A_matrix_ii*charge[:,1].reshape(-1,1)*charge[:,1].reshape(-1,1)
            e_qeq_ii_z= A_matrix_ii*charge[:,2].reshape(-1,1)*charge[:,2].reshape(-1,1)
                
            e_qeq_ii = torch.mean(torch.cat([e_qeq_ii_x,e_qeq_ii_y,e_qeq_ii_z],dim=-1),dim=-1).reshape(-1,1)
            
            e_qeq_i_self = 0.5*e_qeq_ii
            e_qeq_ii = scatter(e_qeq_ii,data['crystal_atom_idx'],dim=0,reduce='sum')
            e_qeq_ii = 0.5*e_qeq_ii
            
            q_chi = torch.mean(charge*data['chi'],dim=-1)
            
            e_qeq_atom=e_qeq_i.squeeze()+e_qeq_i_self.squeeze()+q_chi
            
            e_qeq_ii = e_qeq_ii + scatter(q_chi,data['crystal_atom_idx'],dim=0,reduce='sum').reshape(-1,1)
            
            e_qeq = e_qeq_ij + e_qeq_ii
            data['static_e_atom']=e_qeq_atom
            data['static_e'] = e_qeq
                        
        return data
    
    @staticmethod
    def batch_cal_long(chi):
        chi = chi.reshape(-1, 1).repeat(1, 3)

        return chi
        
    @staticmethod
    def prepare_batch_linear_solver_args(data, chi, A_matrix_ij, A_matrix_ii):
        edge_batch = data['crystal_edge_idx']
        node_batch = data['crystal_atom_idx']
        edge_number = torch.unique(edge_batch, return_counts=True)[1]
        node_number = torch.unique(node_batch, return_counts=True)[1]
        node_edge_idx = data['nbr_fea_idx']
        device=node_edge_idx.device
        node_idx = torch.arange(len(data['atom_fea']),device=device)
        batch_count = len(node_number)
        args = [(batch_number, node_number, edge_number, node_edge_idx, node_idx, A_matrix_ij, A_matrix_ii, chi,device)
                for batch_number in range(batch_count)]
        return args
    

    @staticmethod
    def batch_linear_solver(args,total_charge=0):
        """to solve o3.linear equation for each batch data
            AX=B A is charge interaction of each nodes, B is effective electrongetivity of each nodes, X is charge of each nodes

        Args:
            args (batch_number, node_number, edge_number, node_edge_idx, node_idx, A_matrix_ij, A_matrix_ii, chi): 
                batch_number: batch number of data
                node_number: the number of nodes in each batch
                edge_number: the number of edges in each batch
                node_edge_idx: node index tha compose each edge
                node_idx: node index 
                A_matrix_ij: A matrix of each edge
                A_matrix_ii: A matrix of each node
                chi: effective electronegtivity of each node
                device:device

        Returns:
            data: predicted charge
        """
        batch_number, node_number, edge_number, node_edge_idx, node_idx, A_matrix_ij, A_matrix_ii, chi,device = args
        size=(node_number[batch_number]+1,node_number[batch_number]+1)
        A_matrix=torch.ones(size,device=device)
        A_matrix[-1,-1]=0
        size=(node_number[batch_number],node_number[batch_number])
        if batch_number==0:
            start_node=0
            start_edge=0
        else:
            start_node = torch.sum(node_number[:batch_number])
            start_edge = torch.sum(edge_number[:batch_number])
        edge_end = edge_number[batch_number]+start_edge
        node_end = node_number[batch_number]+start_node
        batch_node_edge_idx=node_edge_idx[start_edge:edge_end].T
        batch_node_idx=node_idx[start_node:node_end].repeat(2).reshape(2,-1)
        total_A_matrix_idx=torch.cat([batch_node_edge_idx,batch_node_idx],dim=1)
        total_A_matrix_idx=total_A_matrix_idx-total_A_matrix_idx.min()
        batch_A_matrix_ij=A_matrix_ij[start_edge:edge_end]
        batch_A_matrix_ii=A_matrix_ii[start_node:node_end]
        batch_A_matrix=torch.cat([batch_A_matrix_ij,batch_A_matrix_ii],dim=0)
        A_sparse = torch.sparse_coo_tensor(total_A_matrix_idx, batch_A_matrix.squeeze(), size=size)
        A_sparse=A_sparse.to_dense().float()
        A_matrix[:size[0],:size[1]]=A_sparse   
        batch_chi=chi[start_node:node_end]
        if len(batch_chi.shape) !=1:
            total_charge=torch.tensor(total_charge,device=device,dtype=torch.float).repeat(batch_chi.shape[-1]).reshape(-1,batch_chi.shape[-1])
        else:
            total_charge=torch.tensor(total_charge,device=device,dtype=torch.float).reshape(-1)
            
        batch_chi=torch.cat([batch_chi,total_charge],dim=0).to(device)
        partial_charge=torch.linalg.solve(A_matrix,batch_chi)
        return partial_charge[:-1]