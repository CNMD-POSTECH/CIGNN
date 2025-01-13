import torch
import torch.nn as nn
from torch_scatter import scatter
from .basic_layers import Dense, ResidualLayers, Crystal_Norm

class modi_cgcnn(nn.Module):
    """
    atom fea를 nbr_atom과 nbr_atom을 연결하는 nbr_fea로 update 기존 cgcnn 그대로 사용
    """
    def __init__(self, atom_fea_len, nbr_fea_len,num_radial ):
        super(modi_cgcnn, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.num_radial = num_radial
        self.nbr_fea_len = nbr_fea_len
        self.fc_full_edge = Dense(in_features=2 * self.atom_fea_len + self.nbr_fea_len,
                                  out_features=2 * self.atom_fea_len,
                                  activation=None,bias=False)
        self.mask_nn = Dense(in_features=self.atom_fea_len,
                        out_features=1,activation='sigmoid',bias=False) #기대하는 것은 값이 큰것은 1에 가깝게 작은것은 0에 가깝게
        self.node_residual = ResidualLayers(n_layers=2,
                                            n_in=self.atom_fea_len,
                                            n_mid=int(self.atom_fea_len/2),
                                            activation='relu')
        self.rbf_emb = Dense(in_features=self.num_radial,
                             out_features=self.nbr_fea_len,
                             activation=None,bias=False)
        
        
        self.atom_norm1 = Crystal_Norm(2*self.atom_fea_len)
        self.atom_norm2 = Crystal_Norm(self.atom_fea_len)
        self.relu1 = nn.ReLU() #양수값은 그대로 가져옴
        self.relu2 = nn.ReLU() #양수값은 그대로 가져옴
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5

    def forward(self, data):
        atom_fea=data['atom_fea']
        nbr_fea=data['edge']
        crystal_index=data['crystal_atom_idx']
        crystal_edge_index=data['crystal_edge_idx']
        nbr_fea_idx = data['nbr_fea_idx']
        rbf= data['rbf']
        rbf_h = self.rbf_emb(rbf) #(N_edge,nbr_fea_len)
        sum_idx=nbr_fea_idx[:,0]
        nbr_fea = nbr_fea * rbf_h
        atom_nbr_fea = atom_fea[nbr_fea_idx].reshape(-1,2*self.atom_fea_len) #(N_edge,2*atom_fea_len)
        total_nbr_fea = torch.cat([atom_nbr_fea, nbr_fea], dim=1) #(N_edge,2*atom_fea+nbr_fea+nbr_fea_len)
        total_gated_nbr_fea = self.fc_full_edge(total_nbr_fea) #(N_edge,2*atom_fea_len)
        total_gated_nbr_fea = self.atom_norm1(total_gated_nbr_fea,crystal_edge_index)
        nbr_filter, nbr_core = total_gated_nbr_fea.chunk(2, dim=1) #(N_edge,atom_fea)*2
        nbr_filter = self.mask_nn(nbr_filter) #(N_edge,1) 
        nbr_core = self.relu1(nbr_core) #(N_edge,atom_fea)
        nbr_sumed = scatter(nbr_filter * nbr_core, sum_idx, dim=0, reduce='mean') #(N_atom,atom_fea) #mean으로 해서 더해 줘야 멀리 있는 것은 덜 가져오게 된다. 
        nbr_sumed = self.atom_norm2(nbr_sumed,crystal_index)
        nbr_sumed = self.node_residual(nbr_sumed) ######      
        data['atom_fea'] = self.inv_sqrt_2*self.relu2(atom_fea + nbr_sumed) #(atom,atom_fea)   ###diffpooling ??????
        if torch.any(torch.isnan(data['atom_fea'])):
            print(torch.where(torch.isnan(data['atom_fea']))[0])
            print('nan')
        return data
    
class modi_cgcnn_edge(nn.Module):
    """
    nbr_fea를 nbr_fea가 연결해주는 atom_fea를 이용하여 update
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(modi_cgcnn_edge, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full_edge = Dense(in_features= self.atom_fea_len + self.nbr_fea_len,
                            out_features=2 * self.nbr_fea_len,
                            activation=None,bias=False)
        self.mask_nn = Dense(in_features= self.nbr_fea_len, 
                             out_features=1,activation='tanh',bias=False)
        self.edge_residual = ResidualLayers(n_layers=2,
                                            n_in=self.nbr_fea_len,
                                            n_mid=int(self.nbr_fea_len/2),
                                            activation='relu')
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.edge_norm1 = Crystal_Norm(2*self.nbr_fea_len)
        self.edge_norm2 = Crystal_Norm(self.nbr_fea_len)
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5

    def forward(self, data):
        atom_fea=data['atom_fea']
        nbr_fea=data['edge']
        crystal_index=data['crystal_atom_idx']
        crystal_edge_index=data['crystal_edge_idx']
        nbr_fea_idx = data['nbr_fea_idx']
        rbf_h = data['rbf']
        # node based edge feature update
        atom_nbr_fea = atom_fea[nbr_fea_idx] #(N_edge,2,atom_fea_len)
        total_nbr_fea = torch.cat([atom_nbr_fea[:,1,:].reshape(-1,self.atom_fea_len)-atom_nbr_fea[:,0,:].reshape(-1,self.atom_fea_len)
                                   , nbr_fea], dim=1) #(N_edge,2*atom_fea+nbr_fea+nbr_fea_len)]])
        total_gated_nbr_fea = self.fc_full_edge(total_nbr_fea) #(N_edge,2*nbr_fea_len)
        total_gated_nbr_fea = self.edge_norm1(total_gated_nbr_fea,crystal_edge_index)
        #total_gated_nbr_fea = self.edge_norm(total_gated_nbr_fea,crystal_edge_index)
        nbr_core, nbr_filter = torch.chunk(total_gated_nbr_fea, 2, dim=1) #(N_edge,nbr_fea_len)*2
        nbr_filter = self.mask_nn(nbr_filter) #(N_edge,nbr_fea_len) # nbr데이터 값이 0~1사이로 나옴 값이 큰것은 중요한 정보를 가지고 있음
        nbr_core = self.relu(nbr_core) #(N_edge,nbr_fea_len)
        nbr_sumed = nbr_filter * nbr_core #(N_edge,nbr_fea_len)
        nbr_sumed = self.edge_norm2(nbr_sumed,crystal_edge_index)
        nbr_sumed = self.edge_residual(nbr_sumed) #(N_edge,nbr_fea_len) #####
        data['edge'] = self.inv_sqrt_2*self.relu(nbr_fea + nbr_sumed) #(N_edge,nbr_fea_len)
        if torch.any(torch.isnan(data['edge'])):
            print('nan')
        return data

class modi_cgcnn_angle(nn.Module):
    """
    angle_fea를 주위의 nbr_fea와 두 nbr_fea가 이루는 angle_fea를 이용하여 angle_ update
    """

    def __init__(self, nbr_fea_len, angle_fea_len):
        super(modi_cgcnn_angle, self).__init__()
        self.nbr_fea_len = nbr_fea_len
        self.angle_fea_len = angle_fea_len
        self.fc_full_edge = Dense(in_features=2 * self.nbr_fea_len + self.angle_fea_len,
                                  out_features=2 * self.angle_fea_len,
                                  activation=None,bias=False)
        self.mask_nn = Dense(in_features= self.angle_fea_len,
                                out_features=1,activation='tanh',bias=False) 
        self.angle_residual = ResidualLayers(n_layers=2,
                                            n_in=self.angle_fea_len,
                                            n_mid=int(self.angle_fea_len/2),
                                            activation='relu')
        self.relu = nn.ReLU()
        self.angle_norm1 = Crystal_Norm(2*self.angle_fea_len)
        self.angle_norm2 = Crystal_Norm(self.angle_fea_len)
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5

    def forward(self, data):
        nbr_fea=data['edge']
        angle_fea=data['angle']
        angle_nbr_idx=data['angle_nbr_idx']
        crystal_angle_index=data['crystal_angle_idx']
        angle_nbr_fea = nbr_fea[angle_nbr_idx].reshape(-1, 2*self.nbr_fea_len) # (N_angle, 2*nbr_fea_len)
        total_angle_fea = torch.cat([angle_fea, angle_nbr_fea], dim=1) #(N_angle,2*nbr_fea_len+angle_fea_len)
        total_gated_angle_fea = self.fc_full_edge(total_angle_fea) #(N_angle,2*angle_fea_len)
        total_gated_angle_fea= self.angle_norm1(total_gated_angle_fea,crystal_angle_index)
        angle_core, angle_filter = torch.chunk(total_gated_angle_fea, 2, dim=1) #(N_angle,angle_fea_len)*2
        angle_core = self.relu(angle_core) # (N_angle,angle_fea_len)
        angle_filter = self.mask_nn(angle_filter) # (N_angle,angle_fea_len)
        angle_sumed = angle_filter * angle_core # (N_angle,angle_fea_len)
        angle_sumed = self.angle_norm2(angle_sumed,crystal_angle_index)
        angle_sumed = self.angle_residual(angle_sumed) # (N_angle,angle_fea_len) #####
        angle_fea = self.inv_sqrt_2*self.relu(angle_fea + angle_sumed) # (N_angle,angle_fea_len)
        data['angle'] = angle_fea
        return data

class modi_cgcnn_a2e(nn.Module):
    """
    nbr_fea를 주위의 nbr_fea와 두 nbr_fea가 이루는 angle_fea를 이용하여 nbr_ update
    """

    def __init__(self, nbr_fea_len, angle_fea_len):
        super(modi_cgcnn_a2e, self).__init__()
        self.nbr_fea_len = nbr_fea_len
        self.angle_fea_len = angle_fea_len
        self.fc_full_edge = Dense(in_features=2 * self.nbr_fea_len + self.angle_fea_len,
                                  out_features=2 * self.nbr_fea_len,
                                  activation=None,bias=False)
        self.mask_nn = Dense(in_features= self.nbr_fea_len,
                                out_features=1,activation='tanh',bias=False)
        self.nbr_residual = ResidualLayers(n_layers=2,
                                            n_in=self.nbr_fea_len,
                                            n_mid=int(self.nbr_fea_len/2),
                                            activation='relu')
        self.relu = nn.ReLU()
        self.angle_norm1 = Crystal_Norm(2*self.nbr_fea_len)
        self.angle_norm2 = Crystal_Norm(self.nbr_fea_len)
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5

    def forward(self,data):
        nbr_fea=data['edge']
        angle_fea=data['angle']
        angle_nbr_idx=data['angle_nbr_idx']
        crystal_edge_idx=data['crystal_edge_idx']
        crystal_angle_index=data['crystal_angle_idx']
        source_idx=angle_nbr_idx[:,0].reshape(-1)
        angle_nbr_fea = nbr_fea[angle_nbr_idx].reshape(-1, 2*self.nbr_fea_len) # (N_angle, 2*nbr_fea_len)
        total_angle_fea = torch.cat([angle_fea, angle_nbr_fea], dim=1) #(N_angle,2*nbr_fea_len+angle_fea_len)
        total_gated_angle_fea = self.fc_full_edge(total_angle_fea) #(N_angle,2*nbr_fea_len)
        total_gated_angle_fea= self.angle_norm1(total_gated_angle_fea,crystal_angle_index)
        nbr_core, nbr_filter = torch.chunk(total_gated_angle_fea, 2, dim=1) #(N_angle,nbr_fea_len)*2
        nbr_core = self.relu(nbr_core) # (N_angle,nbr_fea_len)
        nbr_filter = self.mask_nn(nbr_filter) # (N_angle,nbr_fea_len)
        nbr_sumed = scatter(nbr_filter * nbr_core,source_idx,dim=0,reduce='mean') #(N_nbr,nbr_fea_len)
        if nbr_sumed.shape[0] < crystal_edge_idx.shape[0]:
            nbr_sumed=torch.nn.functional.pad(nbr_sumed,(0,0,0,crystal_edge_idx.shape[0]-nbr_sumed.shape[0]),
                                                mode='constant',
                                                value=0)
        nbr_sumed = self.angle_norm2(nbr_sumed,crystal_edge_idx)
        nbr_sumed = self.nbr_residual(nbr_sumed) # (N_nbr,nbr_fea_len) #####
        nbr_fea = self.inv_sqrt_2*self.relu(nbr_fea + nbr_sumed)
        data['edge'] =nbr_fea
        return data