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


# def atom_distance(
#         positions,
#         nbr_fea_idx,
#         crystal_edge_idx,
#         nbr_off_set,
#         cell):
#     """
#     Compute the distance between atoms in the neighbor list.
#     :param crystal_edge_idx: edge index of the crystal
#     :param positions: (torch.Tenor) ((N_b x N_at) x 3) tensor of atomic positions
#     :param nbr_fea_idx:((N_b x N_at) x max_num_nbr*2,2) tensor of neighbor index
#     :param nbr_off_set:((N_b x N_at) x max_num_nbr x 3)
#     :param cell:(N_b x 3 x 3)
#     :return:
#     """
#     center,target = torch.chunk(nbr_fea_idx, 2, dim=1)
#     center_pos_xyz=positions[center].squeeze()
#     nbr_pos_xyz = positions[target].squeeze()
    
#     edge_per_crystal = torch.unique(crystal_edge_idx, return_counts=True)[1]
#     target_cell=torch.repeat_interleave(cell,edge_per_crystal,dim=0)
#     target_off_set=nbr_off_set.unsqueeze(1).matmul(target_cell).squeeze()
#     target_nbr_pos_xyz =nbr_pos_xyz+target_off_set

#     # target_nbr_pos_list = []
#     # for i in range(len(crystal_edge_idx)):
#     #     target_cell = cell[i]
#     #     target_off_set = nbr_off_set[crystal_edge_idx[i]].matmul(target_cell)
#     #     target_nbr_pos_xyz = nbr_pos_xyz[crystal_edge_idx[i]]+target_off_set
#     #     target_nbr_pos_list.append(target_nbr_pos_xyz)
#     # target_nbr_pos = torch.cat(target_nbr_pos_list, dim=0)
#     # dist_vec = target_nbr_pos - center_pos_xyz
#     dist_vec = target_nbr_pos_xyz - center_pos_xyz
#     distance = torch.norm(dist_vec, dim=1, keepdim=True)
#     if torch.any(distance == 0):
#         raise ValueError('Zero distance between atoms')
#     unit_vec = dist_vec / distance
#     return distance.reshape(-1), unit_vec #(N_edge,),(N_edge,3)

# # def atom_distance(
# #         positions,
# #         nbr_fea_idx,
# #         crystal_atom_idx: list,
# #         nbr_off_set,
# #         cell):
# #     """
# #     Compute the distance between atoms in the neighbor list.
# #     :param crystal_atom_idx: atom index of the crystal
# #     :param positions: (torch.Tenor) ((N_b x N_at) x 3) tensor of atomic positions
# #     :param nbr_fea_idx:((N_b x N_at) x max_num_nbr)
# #     :param nbr_off_set:((N_b x N_at) x max_num_nbr x 3)
# #     :param cell:(N_b x 3 x 3)
# #     :return:
# #     """
# #     N, M = nbr_fea_idx.shape
# #     nbr_pos_xyz = positions[nbr_fea_idx]
# #     ex_pos_xyz = positions.unsqueeze(1).expand(N, M, -1)
# #     total_off_set_list = []
# #     for i in range(len(crystal_atom_idx)):
# #         target_cell = cell[i]
# #         target_index = crystal_atom_idx[i]
# #         target_off_set = nbr_off_set[target_index]
# #         total_off_set_list.append(target_off_set.matmul(target_cell))
# #     total_off_set = torch.cat(total_off_set_list)
# #     t_nbr_pos_xyz = nbr_pos_xyz + total_off_set
# #     dist_vec = t_nbr_pos_xyz - ex_pos_xyz
# #     distance = torch.norm(dist_vec, dim=2, keepdim=True)
# #     unit_vec = dist_vec / distance
# #     return distance, unit_vec

# # def get_spherical(positions,
# #         nbr_fea_idx,
# #         crystal_atom_idx: list,
# #         nbr_off_set,
# #         cell):

# #     N, M = nbr_fea_idx.shape
# #     nbr_pos_xyz = positions[nbr_fea_idx]
# #     ex_pos_xyz = positions.unsqueeze(1).expand(N, M, -1)
# #     total_off_set_list = []
# #     for i in range(len(crystal_atom_idx)):
# #         target_cell = cell[i]
# #         target_index = crystal_atom_idx[i]
# #         target_off_set = nbr_off_set[target_index]
# #         total_off_set_list.append(target_off_set.matmul(target_cell))
# #     total_off_set = torch.cat(total_off_set_list)
# #     t_nbr_pos_xyz = nbr_pos_xyz + total_off_set
# #     xyz = t_nbr_pos_xyz - ex_pos_xyz
# #     r=torch.sqrt(xyz[:,:,0]**2+xyz[:,:,1]**2+xyz[:,:,2]**2)
# #     theta=torch.atan2(torch.sqrt(xyz[:,:,0]**2+xyz[:,:,1]**2),xyz[:,:,2])
# #     phi=torch.atan2(xyz[:,:,1],xyz[:,:,0])
# #     arr=torch.stack([r,theta,phi],dim=2)
# #     return arr

# # class AtomRadial(nn.Module):
# #     def __init__(self):
# #         super(AtomRadial, self).__init__()
        
# #     def forward(self, positions, nbr_fea_idx, crystal_atom_idx, nbr_off_set, cell):

# #        return get_spherical(positions, nbr_fea_idx, crystal_atom_idx, nbr_off_set, cell)
       




# def get_angle(
#         position,
#         angle_fea_idx,
#         crystal_angle_idx: list,
#         angle_off_set,
#         cell):
#     """

#     :param position: (torch.Tenor) ((N_b x N_at) x 3) tensor of atomic positions
#     :param angle_fea_idx:
#     :param crystal_atom_idx:
#     :param angle_off_set:
#     :param cell:
#     :return:
#     """
#     angle_pos_xyz = position[angle_fea_idx]
#     # total_angle_off_set_list_b,total_angle_off_set_list_c = [],[]
#     # for i in range(len(crystal_angle_idx)):
#     #     target_cell = cell[i]
#     #     target_index = crystal_angle_idx[i]
#     #     target_off_set_b,target_off_set_c = angle_off_set[target_index].chunk(2, dim=1)
#     #     total_angle_off_set_list_b.append(target_off_set_b.squeeze().matmul(target_cell))
#     #     total_angle_off_set_list_c.append(target_off_set_c.squeeze().matmul(target_cell))
    
#     # target_off_set_b = torch.cat(total_angle_off_set_list_b)
#     # target_off_set_c = torch.cat(total_angle_off_set_list_c)
#     # a,b,c = torch.chunk(angle_pos_xyz, 3, dim=1)
#     #torch.save(angle_pos_xyz,'/home/idealhun92/MLMD/position.pt')
#     angle_per_crystal = torch.unique(crystal_angle_idx, return_counts=True)[1]
#     target_cell_angle=torch.repeat_interleave(cell,angle_per_crystal,dim=0)
#     target_off_set_angle= angle_off_set.matmul(target_cell_angle)
#     b,c=torch.chunk(angle_pos_xyz[:,1:,:]+target_off_set_angle,2,dim=1)
#     a=angle_pos_xyz[:,0,:] # (N_angle,3)
#     b=b.squeeze() # (N_angle,3)
#     c=c.squeeze() # (N_angle,3)
#     # a=a.squeeze()
#     # b=b.squeeze()+ target_off_set_b
#     # c=c.squeeze()+ target_off_set_c
#     ## a-b-c
#     vector1 = a - b # (N_b x N_at) x 3
#     vector2 = c - b # (N_b x N_at) x 3
#     ##if b-a-c 
#     # vector1 = b - a
#     # vector2 = c - a
#     #vector3 = (two - one).squeeze(2)
#     r1c = torch.norm(vector1, dim=1)
#     if torch.any(r1c == 0):
#         raise ValueError('Zero distance between atoms')
#     r2c = torch.norm(vector2, dim=1)
#     if torch.any(r2c == 0):
#         raise ValueError('Zero distance between atoms')
#     # cos(alpha) = (u * v) / (|u|*|v|)
#     x = torch.sum(vector1 * vector2, dim=1) #(N_at x max_num_angle)
#      # sin(alpha) = |u x v| / (|u|*|v|)
#     y = torch.cross(vector1,vector2).norm(dim=-1) # (N_at x max_num_angle,1)
#     y = torch.max(y,torch.tensor(1e-9))
#     angle_radi = torch.atan2(y,x)

#     # return r1c, r2c, angle_radi # (N_angle,), (N_angle,), (N_angle,)
#     return angle_radi


# class AtomAngle(nn.Module):
#     """
#     Compute the distance between atoms in the neighbor list.
#     """

#     def __init__(self):
#         super(AtomAngle, self).__init__()

#     def forward(self, positions, angle_fea_idx, crystal_atom_idx, angle_off_set, cell):
#         return get_angle(positions, angle_fea_idx, crystal_atom_idx, angle_off_set, cell)


# # def get_dihedral(
# #         position,
# #         dih_fea_idx,
# #         crystal_atom_idx: list,
# #         dih_fea_off_set,
# #         cell):
# #     dihedral_pos_xyz = position[dih_fea_idx]
# #     total_dih_off_set_list = []
# #     for i in range(len(crystal_atom_idx)):
# #         target_cell = cell[i]
# #         target_index = crystal_atom_idx[i]
# #         target_off_set = dih_fea_off_set[target_index]
# #         total_dih_off_set_list.append(target_off_set.matmul(target_cell))
# #     total_dih_off_set = torch.cat(total_dih_off_set_list)
# #     true_pos_xyz = dihedral_pos_xyz + total_dih_off_set
# #     center, one, two, three = torch.chunk(true_pos_xyz, 4, dim=2)
# #     vector1 = (center - one).squeeze(2)
# #     vector2 = (two - center).squeeze(2)
# #     vector3 = (three - two).squeeze(2)
# #     vector4 = (three - one).squeeze(2)
# #     rc1 = torch.norm(vector1, dim=2)
# #     r2c = torch.norm(vector2, dim=2)
# #     r32 = torch.norm(vector3, dim=2)
# #     r31 = torch.norm(vector4, dim=2)
# #     v23 = torch.cross(vector2, vector3)
# #     v12 = torch.cross(vector1, vector2)
# #     dih_rad = torch.atan2(torch.norm(vector2, dim=2) * torch.sum((vector1 * v23), dim=2), torch.sum((v12 * v23), dim=2))
# #     dih_degree = torch.rad2deg(
# #         torch.atan2(torch.norm(vector2, dim=2) * torch.sum((vector1 * v23), dim=2), torch.sum((v12 * v23), dim=2)))
# #     return rc1, r2c, r32, r31, dih_rad, dih_degree


# # class AtomDihedral(nn.Module):
# #     """
# #     Compute the distance between atoms in the neighbor list.
# #     """

# #     def __init__(self):
# #         super(AtomDihedral, self).__init__()

# #     def forward(self, positions, dih_fea_idx, crystal_atom_idx, dih_fea_off_set, cell):
# #         return get_dihedral(positions, dih_fea_idx, crystal_atom_idx, dih_fea_off_set, cell)
