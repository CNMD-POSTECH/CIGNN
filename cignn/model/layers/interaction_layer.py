import torch
import torch.nn as nn
from cignn.model.layers.basic_layers import *
from cignn.model.layers.feature_update_layer import *

class InteractionLayer(nn.Module):
    def __init__(self,
                 atom_fea_len,
                 nbr_fea_len,
                 angle_fea_len,
                 num_radial,
                 h_fea_len,
                 n_h,
                 charge=False,
                 edge_swap=False,
                 dropout=0.0,
                 element_len=2):
        self.num_radial=num_radial
        self.edge_swap=edge_swap
        self.atom_fea_len=atom_fea_len
        self.nbr_fea_len=nbr_fea_len
        self.angle_fea_len=angle_fea_len
        self.h_fea_len=h_fea_len
        self.n_h=n_h
        self.dropout=dropout
        self.charge=charge
        self.element_len=element_len
        super(InteractionLayer, self).__init__()
        
        ### Convolution layers
        self.e2n_1=modi_cgcnn(atom_fea_len=self.atom_fea_len,
                              nbr_fea_len=self.nbr_fea_len,
                              num_radial=self.num_radial,
                              charge=self.charge,
                              element_len=self.element_len)
        self.e2n_2=modi_cgcnn(atom_fea_len=self.atom_fea_len,
                              nbr_fea_len=self.nbr_fea_len,
                              num_radial=self.num_radial,
                              charge=self.charge,
                              element_len=self.element_len) 
        self.n2e_1=modi_cgcnn_edge(atom_fea_len=self.atom_fea_len,
                                   nbr_fea_len=self.nbr_fea_len,
                                   charge=self.charge) 
        self.e2a_1=  modi_cgcnn_angle(nbr_fea_len=self.nbr_fea_len,
                                      angle_fea_len=self.angle_fea_len) 
        self.a2e_1= modi_cgcnn_a2e(nbr_fea_len=self.nbr_fea_len, 
                                   angle_fea_len=self.angle_fea_len) 
        # residual layer
        self.node_residual1=ResidualLayers(self.n_h,self.atom_fea_len,int(self.atom_fea_len/2)) 

        self.node_residual2=ResidualLayers(self.n_h,self.atom_fea_len,int(self.atom_fea_len/2)) 
                                           
        self.edge_residual1=ResidualLayers(self.n_h,self.nbr_fea_len,int(self.nbr_fea_len/2)) 
                                           
        self.edge_residual2=ResidualLayers(self.n_h,self.nbr_fea_len,int(self.nbr_fea_len/2)) 
                                           
        self.edge_residual3=ResidualLayers(self.n_h,self.nbr_fea_len,int(self.nbr_fea_len/2)) 
                                          
        self.angle_residual1=ResidualLayers(self.n_h,self.angle_fea_len,int(self.angle_fea_len/2)) 
                                           
        self.angle_residual2=ResidualLayers(self.n_h,self.angle_fea_len,int(self.angle_fea_len/2))
                                        
        self.inv_sqrt_2=1 / (2.0) ** 0.5
        
    def forward(self , data):
            atom_fea=data['atom_fea']
            edge_fea=data['edge']
            angle_fea=data['angle']
            crystal_edge_idx=data['crystal_edge_idx']
            crystal_angle_idx=data['crystal_angle_idx']
            nbr_fea_idx=data['nbr_fea_idx']
            nbr_swap_idx=data['nbr_swap_idx']
            angle_nbr_idx=data['angle_nbr_idx']
            rbf=data['rbf']
            atom_attr=data['atom_attr']
            if 'q_fea' in data.keys():
                q_fea=data['q_fea']
                nbr_vector=data['nbr_vec']
            else:
                q_fea=None
                nbr_vector=None
                
        #node to edge
            edg_ori=edge_fea
            edge_fea=self.n2e_1(atom_fea,edge_fea,nbr_fea_idx,crystal_edge_idx,q_fea,nbr_vector)
            edge_fea=self.edge_residual1(edge_fea)
            edge_fea=self.inv_sqrt_2*(edge_fea + edg_ori)

        # edge to angle
            angle_ori=angle_fea
            angle_fea=self.e2a_1(edge_fea,angle_fea,angle_nbr_idx,crystal_angle_idx)
            angle_fea=self.angle_residual2(angle_fea)
            angle_fea=self.inv_sqrt_2*(angle_fea+angle_ori)

        #angle to edge
            edge_ori=edge_fea
            edge_fea=self.a2e_1(edge_fea,angle_fea,angle_nbr_idx,crystal_edge_idx,crystal_angle_idx)     
            edge_fea=self.edge_residual3(edge_fea)
            edge_fea=self.inv_sqrt_2*(edge_fea+edge_ori)

        # edge to node
            atom_ori=atom_fea
            atom_fea=self.e2n_1(atom_fea,edge_fea,rbf,nbr_fea_idx,crystal_edge_idx,atom_attr,q_fea)
            atom_fea=self.node_residual1(atom_fea)
            atom_fea=self.inv_sqrt_2*(atom_fea+atom_ori)

        ##swap###
            if self.edge_swap:
                edge_swap=edge_fea[nbr_swap_idx] # (n_edge,fea)
                edge_fea=torch.mean(torch.concat([edge_fea.unsqueeze(1),edge_swap.unsqueeze(1)],dim=1),dim=1) # (n_edge,fea)
            atom_ori=atom_fea
            atom_fea=self.e2n_2(atom_fea,edge_fea,rbf,nbr_fea_idx,crystal_edge_idx,atom_attr,q_fea)
            atom_fea=self.node_residual2(atom_fea)
            atom_fea=self.inv_sqrt_2*(atom_fea+atom_ori)
 
            data['atom_fea']=atom_fea
            data['edge']=edge_fea
            data['angle']=angle_fea
            return data