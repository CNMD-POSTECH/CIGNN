import torch
import torch.nn as nn
from torch_scatter import scatter
from .basic_layers import Dense, ResidualLayers, Crystal_Norm

class modi_cgcnn(nn.Module):
    """
    atom fea를 nbr_atom과 nbr_atom을 연결하는 nbr_fea로 update 기존 cgcnn 그대로 사용
    """
    def __init__(self, atom_fea_len, nbr_fea_len,num_radial, charge=False, element_len=2):
        super(modi_cgcnn, self).__init__()
        self.num_radial = num_radial
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.charge=charge
        self.element_len = element_len
        self.fc_full_edge = Dense(in_features=2 * self.atom_fea_len + self.nbr_fea_len,
                                  out_features=2 * self.atom_fea_len,
                                  activation=None,bias=False)
        self.mask_nn = Dense(in_features=self.atom_fea_len,
                        out_features=1,activation='sigmoid',bias=False) #기대하는 것은 값이 큰것은 1에 가깝게 작은것은 0에 가깝게

        self.rbf_emb = Dense(in_features=self.num_radial,
                             out_features=self.nbr_fea_len,
                             activation=None,bias=False)
        self.atom_norm1 = Crystal_Norm(2*self.atom_fea_len)
        self.silu1 = nn.SiLU() #양수값은 그대로 가져옴
        self.silu2 = nn.SiLU() #양수값은 그대로 가져옴
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5
        self.sc = Dense(in_features=element_len,
                        out_features=self.atom_fea_len,
                        activation=None,bias=False)

        self.atom_norm_filter = Crystal_Norm(self.atom_fea_len)
        self.atom_norm_core = nn.LayerNorm(self.atom_fea_len)
        
        if self.charge == True:
            self.charge_emb=Dense(in_features=self.atom_fea_len,
                                  out_features=self.atom_fea_len,activation=None,bias=False)
            self.inv_sqrt_3=1 / (3.0) ** 0.5
            self.proj = Dense(in_features=3*self.atom_fea_len,
                                    out_features=self.atom_fea_len,activation=None,bias=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_nn.weight)
        
    def forward(self, atom_fea, nbr_fea,rbf, nbr_fea_idx,crystal_edge_index,atom_attr,q_fea=None):
        if q_fea != None and self.charge == True:
            charge_emb=self.charge_emb(q_fea).reshape(-1,3*self.atom_fea_len)
            charge_emb = self.proj(charge_emb)
        sum_idx=nbr_fea_idx[:,0]
        sc = self.sc(atom_attr)
        nbr_fea = nbr_fea * self.rbf_emb(rbf) #(N_edge,nbr_fea_len)
        atom_nbr_fea = atom_fea[nbr_fea_idx].reshape(-1,2*self.atom_fea_len) #(N_edge,2*atom_fea_len)
        total_nbr_fea = torch.cat([atom_nbr_fea, nbr_fea], dim=1) #(N_edge,2*atom_fea+nbr_fea+nbr_fea_len)
        total_gated_nbr_fea = self.fc_full_edge(total_nbr_fea) #(N_edge,2*atom_fea_len)
        total_gated_nbr_fea = self.atom_norm1(total_gated_nbr_fea,crystal_edge_index) #(N_edge,2*atom_fea_len)
        
        nbr_filter, nbr_core = total_gated_nbr_fea.chunk(2, dim=1) #(N_edge,atom_fea)*2
        nbr_filter = self.mask_nn(nbr_filter) #(N_edge,1) 
        
        nbr_core = self.atom_norm_core(nbr_core) #(N_edge,atom_fea)
        nbr_core = self.silu1(nbr_core) #(N_edge,atom_fea)
        nbr_sumed = scatter(nbr_filter * nbr_core, sum_idx, dim=0, reduce='sum') #(N_atom,atom_fea) #mean으로 해서 더해 줘야 멀리 있는 것은 덜 가져오게 된다.
        src_index,nbr_counts= sum_idx.unique(return_counts=True)  #(N_atom,1)
        nbr_counts = scatter(nbr_counts, src_index, dim=0, reduce='mean') #(N_atom,1)
        nbr_counts = nbr_counts + 1e-6
        nbr_counts = nbr_counts.sqrt().reshape(-1,1)
        nbr_sumed=nbr_sumed/nbr_counts #(N_atom,atom_fea) 
        # nbr_sumed = scatter(nbr_filter * nbr_core, sum_idx, dim=0, reduce='sum') #(N_atom,atom_fea) #mean으로 해서 더해 줘야 멀리 있는 것은 덜 가져오게 된다.
        nbr_counts_test= sum_idx.unique(return_counts=True)[1].reshape(-1,1) #(N_atom,1)
        nbr_counts_test = nbr_counts_test + 1e-6
        nbr_counts_test = nbr_counts_test.sqrt()
        test=torch.sum(nbr_counts_test-nbr_counts)
        if test > 0 :
            raise ValueError('test')
        
        if q_fea != None and self.charge ==True:
            atom_fea = self.inv_sqrt_3*(nbr_sumed + sc + charge_emb)
        else:
            atom_fea = self.inv_sqrt_2*(nbr_sumed + sc) #(N_atom,atom_fea)
        return atom_fea
class modi_cgcnn_edge(nn.Module):
    """
    nbr_fea를 nbr_fea가 연결해주는 atom_fea를 이용하여 update
    """

    def __init__(self, atom_fea_len, nbr_fea_len,charge=False):
        super(modi_cgcnn_edge, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.charge = charge
        self.fc_full_edge = Dense(in_features= 2*self.atom_fea_len + self.nbr_fea_len,
                            out_features=2 * self.nbr_fea_len,
                            activation=None,bias=False)
        self.mask_nn = Dense(in_features= self.nbr_fea_len, 
                             out_features=1,activation='sigmoid',bias=False)
        self.edge_residual = ResidualLayers(n_layers=2,
                                            n_in=self.nbr_fea_len,
                                            n_mid=int(self.nbr_fea_len/2),
                                            activation='silu')
        self.edge_proj =Dense(in_features= self.nbr_fea_len,
                            out_features=self.nbr_fea_len,
                            activation=None,bias=False)
        self.silu = nn.SiLU()
        self.edge_norm1 = Crystal_Norm(2*self.nbr_fea_len)
        self.edge_norm2 = Crystal_Norm(self.nbr_fea_len)
        self.edge_norm_core = nn.LayerNorm(self.nbr_fea_len)
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5
        if self.charge ==True:
            self.charge_emb = Dense(in_features=2*self.atom_fea_len + self.nbr_fea_len,
                                    out_features = self.nbr_fea_len)
            self.inv_sqrt_3=1 / (3.0) ** 0.5
            self.proj = Dense(in_features=3*self.nbr_fea_len,
                                    out_features=self.nbr_fea_len,activation=None,bias=False)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_nn.weight)
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx,crysta_edge_index,q_fea=None,nbr_vector=None):
        if q_fea != None and self.charge == True:
            q_fea = q_fea[nbr_fea_idx].reshape(-1,3,2*self.atom_fea_len)
            vector= nbr_vector.reshape(-1,3,1)*nbr_fea.unsqueeze(1).repeat(1,3,1)
            total_nbr_q_fea =torch.cat([q_fea,vector],dim=-1)
            charge_emb=self.charge_emb(total_nbr_q_fea)
            charge_emb = self.proj(charge_emb.reshape(-1,3*self.nbr_fea_len))
            # nbr_fea = torch.concat([nbr_fea,charge_emb],dim=-1) #(N_edge,2*nbr_fea_len)
            # nbr_fea = self.proj(nbr_fea)
        
        # node based edge feature update
        atom_nbr_fea = atom_fea[nbr_fea_idx] #(N_edge,2,atom_fea_len)
        total_nbr_fea = torch.cat([atom_nbr_fea.reshape(-1,2*self.atom_fea_len), nbr_fea], dim=1) #(N_edge,2*atom_fea+nbr_fea+nbr_fea_len)]])
        total_gated_nbr_fea = self.fc_full_edge(total_nbr_fea) #(N_edge,2*nbr_fea_len)
        total_gated_nbr_fea = self.edge_norm1(total_gated_nbr_fea,crysta_edge_index)
        
        nbr_core, nbr_filter = torch.chunk(total_gated_nbr_fea, 2, dim=1) #(N_edge,nbr_fea_len)*2
        nbr_filter = self.mask_nn(nbr_filter) #(N_edge,nbr_fea_len) # nbr데이터 값이 0~1사이로 나옴 값이 큰것은 중요한 정보를 가지고 있음
        nbr_core = self.edge_norm_core(nbr_core) #(N_edge,nbr_fea_len)
        nbr_core = self.silu(nbr_core) #(N_edge,nbr_fea_len)
        
        nbr_sumed = nbr_filter * nbr_core #(N_edge,nbr_fea_len)
        
        nbr_sumed = self.edge_norm2(nbr_sumed,crysta_edge_index)
        nbr_sumed = self.edge_residual(nbr_sumed) #(N_edge,nbr_fea_len) #####

        if q_fea != None and self.charge ==True:
            nbr_fea = self.inv_sqrt_3*(nbr_fea + nbr_sumed + charge_emb)
        else:
            nbr_fea = self.inv_sqrt_2*(nbr_fea + nbr_sumed) #(N_edge,nbr_fea_len)
        return nbr_fea

class modi_cgcnn_angle(nn.Module):
    """
    nbr_fea를 주위의 nbr_fea와 두 nbr_fea가 이루는 angle_fea를 이용하여 nbr_ update
    """

    def __init__(self, nbr_fea_len, angle_fea_len):
        super(modi_cgcnn_angle, self).__init__()
        self.nbr_fea_len = nbr_fea_len
        self.angle_fea_len = angle_fea_len
        self.fc_full_edge = Dense(in_features=2 * self.nbr_fea_len + self.angle_fea_len,
                                  out_features=2 * self.angle_fea_len,
                                  activation=None,bias=False)
        self.mask_nn = Dense(in_features= self.angle_fea_len,
                                out_features=1,activation='sigmoid',bias=False) 
        self.angle_residual = ResidualLayers(n_layers=2,
                                            n_in=self.angle_fea_len,
                                            n_mid=int(self.angle_fea_len/2),
                                            activation='silu')
        self.angle_proj = Dense(in_features=self.angle_fea_len,
                            out_features=self.angle_fea_len,
                            activation=None,bias=False)
        self.relu = nn.SiLU()
        self.angle_norm1 = Crystal_Norm(2*self.angle_fea_len)
        self.angle_norm2 = Crystal_Norm(self.angle_fea_len)
        self.angle_norm_core = nn.LayerNorm(self.angle_fea_len)
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5
    
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_nn.weight)

    def forward(self, nbr_fea, angle_fea, angle_nbr_idx,crystal_angle_index):
        angle_nbr_fea = nbr_fea[angle_nbr_idx].reshape(-1, 2*self.nbr_fea_len) # (N_angle, 2*nbr_fea_len)
        total_angle_fea = torch.cat([angle_fea, angle_nbr_fea], dim=1) #(N_angle,2*nbr_fea_len+angle_fea_len)
        total_gated_angle_fea = self.fc_full_edge(total_angle_fea) #(N_angle,2*angle_fea_len)
        total_gated_angle_fea= self.angle_norm1(total_gated_angle_fea,crystal_angle_index)
        
        angle_core, angle_filter = torch.chunk(total_gated_angle_fea, 2, dim=1) #(N_angle,angle_fea_len)*2
        angle_core=self.angle_norm_core(angle_core) #(N_angle,angle_fea_len)
        angle_core = self.relu(angle_core) # (N_angle,angle_fea_len)
        angle_filter = self.mask_nn(angle_filter) # (N_angle,angle_fea_len)
        angle_sumed = angle_filter * angle_core # (N_angle,angle_fea_len)
        angle_sumed = self.angle_norm2(angle_sumed,crystal_angle_index)
        angle_sumed = self.angle_residual(angle_sumed) # (N_angle,angle_fea_len) #####
        angle_fea = self.inv_sqrt_2*(angle_fea + angle_sumed) # (N_angle,angle_fea_len)

        return angle_fea

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
                                out_features=1,activation='sigmoid',bias=False)
        self.nbr_residual = ResidualLayers(n_layers=2,
                                            n_in=self.nbr_fea_len,
                                            n_mid=int(self.nbr_fea_len/2),
                                            activation='silu')
        self.edge_proj = Dense(in_features=self.nbr_fea_len,
                                  out_features=self.nbr_fea_len,
                                  activation=None,bias=False)
        self.silu = nn.SiLU()
        self.angle_norm1 = Crystal_Norm(2*self.nbr_fea_len)
        self.angle_norm2 = nn.LayerNorm(self.nbr_fea_len)
        self.angle_core_norm = nn.LayerNorm(self.nbr_fea_len)
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5
    
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_nn.weight)

    def forward(self, nbr_fea, angle_fea, angle_nbr_idx,crystal_edge_idx,crystal_angle_index):
        source_idx=angle_nbr_idx[:,0].reshape(-1)
        angle_nbr_fea = nbr_fea[angle_nbr_idx].reshape(-1, 2*self.nbr_fea_len) # (N_angle, 2*nbr_fea_len)
        total_angle_fea = torch.cat([angle_fea, angle_nbr_fea], dim=1) #(N_angle,2*nbr_fea_len+angle_fea_len)
        total_gated_angle_fea = self.fc_full_edge(total_angle_fea) #(N_angle,2*nbr_fea_len)
        total_gated_angle_fea= self.angle_norm1(total_gated_angle_fea,crystal_angle_index)
        
        nbr_core, nbr_filter = torch.chunk(total_gated_angle_fea, 2, dim=1) #(N_angle,nbr_fea_len)*2
        nbr_core = self.angle_core_norm(nbr_core) #(N_angle,nbr_fea_len)
        nbr_core = self.silu(nbr_core)# (N_angle,nbr_fea_len)
        nbr_filter = self.mask_nn(nbr_filter) # (N_angle,1)
        nbr_sumed = nbr_filter*nbr_core #(N_angle,2,nbr_fea_len)
        nbr_sumed = scatter(nbr_sumed,source_idx,dim=0,reduce='mean') #(N_nbr,nbr_fea_len)
        if nbr_sumed.shape[0] < crystal_edge_idx.shape[0]:
            nbr_sumed=torch.nn.functional.pad(nbr_sumed,(0,0,0,crystal_edge_idx.shape[0]-nbr_sumed.shape[0]),
                                                mode='constant',
                                                value=0)
        nbr_sumed = self.angle_norm2(nbr_sumed)
        # nbr_sumed = self.edge_proj(nbr_sumed) # (N_nbr,nbr_fea_len) #####
        nbr_sumed = self.nbr_residual(nbr_sumed)
        nbr_fea = self.inv_sqrt_2*(nbr_fea + nbr_sumed)

        return nbr_fea
