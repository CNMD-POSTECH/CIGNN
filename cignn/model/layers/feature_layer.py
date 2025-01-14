import torch
from torch import nn
from e3nn.o3 import spherical_harmonics

class AtomDistance(nn.Module):
    """
    Compute the distance between atoms in the neighbor list.
    Compute the self vector between periodic
    """
    def __init__(self,lmax):
        super(AtomDistance, self).__init__()
        self.lmax = lmax
        self.orbitals = list(range(self.lmax))
    def get_nbr_vec(self,data):
        edge_count=torch.unique(data['crystal_edge_idx'],return_counts=True)[1].reshape(-1)
        center=data['nbr_fea_idx'][:,0]
        nbr= data['nbr_fea_idx'][:,1]
        center_position=data['position'][center]
        nbr_position=data['position'][nbr]
        cells=data['cell'].repeat_interleave(edge_count,dim=0)
        nbr_position += torch.sum(torch.matmul(torch.diag_embed(data['nbr_off_set']),cells),dim=-2)
        nbr_vector=nbr_position-center_position
        nbr_vector=torch.round(nbr_vector * 10**5) / 10**5
        return nbr_vector

    def get_self_vec(self,data):
        atom_count=torch.unique(data['crystal_atom_idx'],return_counts=True)[1].reshape(-1)
        self_cells=torch.matmul(torch.diag_embed(data['i_origin_offset']).to(torch.double),data['cell'][:,None,:,:])
        self_atom_dis=self_cells.repeat_interleave(atom_count,dim=0)
        self_periodic=torch.sum(torch.diag_embed(data['position'])[:,None,:,:]+self_atom_dis,dim=-2)
        self_vector=self_periodic-data['position'][:,None,:]
        return self_vector

    def forward(self, data,self_vector=False):
        edge_count=torch.unique(data['crystal_edge_idx'],return_counts=True)[1].reshape(-1)
        center=data['nbr_fea_idx'][:,0]
        nbr= data['nbr_fea_idx'][:,1]
        center_position=data['position'][center]
        nbr_position=data['position'][nbr]
        cells=data['cell'].repeat_interleave(edge_count,dim=0)
        nbr_position =nbr_position + torch.sum(torch.matmul(torch.diag_embed(data['nbr_off_set']),cells),dim=-2)
        data['nbr_vec']=nbr_position-center_position
        orbitals=spherical_harmonics(self.orbitals,data['nbr_vec'],normalize=False,normalization='component')
        data['orbitals']=orbitals
        data['distance']=torch.norm(data['nbr_vec'],dim=-1)
        data['unit_vec']=data['nbr_vec']/(data['distance'].reshape(-1,1))
        
        if self_vector == True:
          atom_count=torch.unique(data['crystal_atom_idx'],return_counts=True)[1].reshape(-1)
          self_cells=torch.matmul(torch.diag_embed(data['i_origin_offset']).to(torch.double),data['cell'][:,None,:,:])
          self_atom_dis=self_cells.repeat_interleave(atom_count,dim=0)
          self_periodic=torch.sum(torch.diag_embed(data['position'])[:,None,:,:]+self_atom_dis,dim=-2)
          data['self_vec']=self_periodic-data['position'][:,None,:]
          # data['self_vec']=self.get_self_vec(data)
        return data

class AtomAngle(nn.Module):
    """
    Compute the distance between atoms in the neighbor list.
    """

    def __init__(self):
        super(AtomAngle, self).__init__()
    
    def get_angle(self,data):
      angle_vector=data['nbr_vec'][data['angle_nbr_idx']] #[vx1,vy1,vz1,vx2,vy2,vz2]
      vector1=-1*angle_vector[:,0,:]
      vector2=angle_vector[:,1,:]
      if torch.any(torch.norm(vector1, dim=-1)==0) or torch.any(torch.norm(vector2, dim=-1)==0) :
        raise ValueError('Zero distance between atoms')
      x = torch.sum(vector1 * vector2, dim=1) #(N_at x max_num_angle)
      # sin(alpha) = |u x v| / (|u|*|v|)
      y = torch.cross(vector1,vector2).norm(dim=-1) # (N_at x max_num_angle,1)
      y = torch.max(y,torch.tensor(1e-9))
      angle_radi = torch.atan2(y,x)
      return angle_radi

    def forward(self, data):
        data['angle_radi']=self.get_angle(data)
        return data