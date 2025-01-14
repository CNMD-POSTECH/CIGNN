from cignn.model.layers import embedding_layer, feature_layer, basis_layers
from .layers.interaction_layer import InteractionLayer
from .layers.basic_layers import *
from .layers.conv_layer import *
from .layers.qeq_layer import *
import torch

#change.log 2023.02.27
#1. add number_of_atoms in crystal
#2. change crystal_atom_idx,crystal_edge_idx,crystal_angle_idx

def InvCNMD(atom_type,
            atom_fea_len,
            nbr_fea_len,
            angle_fea_len,
            h_fea_len,
            n_conv,
            n_h,
            num_radial,
            num_spherical,
            direct,
            charge,
            energy_mean,
            species_force_rms,
            force_rms,
            q_energy_mean=None,
            cutoff=5,
            element_len=2):
  """
  Parameters
  ----------
  data is composed with 
  atom_species:  # of atoms species
  cgcnn_fea_len: # of original atom feature dim (from CGCNN, atom_init.json)
  atom_fea_len: # of atom feature dim
  nbr_fea_len: # of nbr feature dim
  angle_fea_len: # of angle fea dim
  h_fea_len: # of hideen features
  n_conv: # of interaction layers
  n_h: # of output layers
  num_radial: # of radial functions
  num_spherical: # of spherical functions
  direct: get force as direct or gradient method
  max_nbr: # of nbr atoms
  cutoff: cut off range in nbr is cut of radial function 
        
  """
  if direct==True:
    edge_swap=True
  else:
    edge_swap=False

  layers_list=[]
  layers_list.extend([feature_layer.AtomDistance(lmax=2),
                      feature_layer.AtomAngle(),
                      basis_layers.BesselBasisLayer(num_radial=num_radial, cutoff=cutoff),
                      basis_layers.SphericalBasisLayer(num_spherical=num_spherical,num_radial=num_radial,cutoff=cutoff)
                      ])
  layers_list.extend([embedding_layer.AtomEmbeddingLayer(atom_type=atom_type, 
                                                         atom_fea_len=atom_fea_len,
                                                         charge=charge),
                      
                      embedding_layer.EdgeEmbeddingLayer(atom_fea_len=atom_fea_len,
                                                         nbr_fea_len=nbr_fea_len,
                                                         num_radial=num_radial),

                      embedding_layer.AEEmbeddingLayer(nbr_fea_len=nbr_fea_len,
                                                       out_fea_len=angle_fea_len,
                                                       num_radial=num_radial,
                                                       num_spherical=num_spherical,
                                                      ),

                      OutData(atom_fea_len=atom_fea_len,
                              nbr_fea_len=nbr_fea_len,
                              h_fea_len=h_fea_len,
                              num_radial=num_radial,
                              n_h=n_h,
                              direct=direct,
                              activation='silu')])
  for i in range(n_conv):
    layers_list.extend([InteractionLayer(atom_fea_len=atom_fea_len,
                                        nbr_fea_len=nbr_fea_len,
                                        angle_fea_len=angle_fea_len,
                                        h_fea_len=h_fea_len,
                                        num_radial=num_radial,
                                        n_h=n_h,
                                        charge=charge,
                                        edge_swap=edge_swap,
                                        dropout=0.0,
                                        element_len=element_len)])

    layers_list.extend([OutData(atom_fea_len=atom_fea_len,
                                nbr_fea_len=nbr_fea_len,
                                h_fea_len=h_fea_len,
                                n_h=n_h,
                                direct=direct,
                                num_radial=num_radial,
                                activation='silu')])
  layers_list.extend([ElementScale(atom_type=atom_type,
                                   scale=species_force_rms,
                                   systemE=energy_mean,
                                   QeqE=q_energy_mean,
                                   divider=force_rms
                                   )])
  layers_list.extend([Fin_Out(direct=direct,
                              divider=force_rms
                            )])

  return ListGraphNetwork(layers=layers_list,direct=direct)
                                                                                    
def InvCNMD_Q(atom_type,
              atom_fea_len,
              nbr_fea_len,
              n_conv,
              num_radial,
              lmax,
              direct,
              e_field=[0,0,0],
              cutoff=5):
  """
  Parameters
  ----------
  data is composed with 
  atom_species:  # of atoms species
  cgcnn_fea_len: # of original atom feature dim (from CGCNN, atom_init.json)
  atom_fea_len: # of atom feature dim
  nbr_fea_len: # of nbr feature dim
  angle_fea_len: # of angle fea dim
  h_fea_len: # of hideen features
  n_conv: # of interaction layers
  n_h: # of output layers
  num_radial: # of radial functions
  num_spherical: # of spherical functions
  direct: get force as direct or gradient method
  max_nbr: # of nbr atoms
  cutoff: cut off range in nbr is cut of radial function 
  e_field : start end axis=0,1,2
  """
  layers_list=[]
  
  layers_list.extend([feature_layer.AtomDistance(lmax=lmax),
                      basis_layers.BesselBasisLayer(num_radial=num_radial, cutoff=cutoff)])

  layers_list.extend([embedding_layer.AtomEmbeddingLayer(atom_type=atom_type, 
                                                         atom_fea_len=atom_fea_len),

                      embedding_layer.VectorEmbeddinglayer(atom_fea_len=atom_fea_len,
                                                           nbr_fea_len=nbr_fea_len,
                                                           lmax=2,
                                                           num_radial=num_radial,
                                                           activation='relu'),

                      embedding_layer.RbfEmb(num_radial=num_radial,
                                             nbr_fea_len=nbr_fea_len)])
  for i in range(n_conv):
    layers_list.extend([modi_cgcnn(atom_fea_len=atom_fea_len,
                                   nbr_fea_len=nbr_fea_len,
                                   num_radial=num_radial,)])
    layers_list.extend([modi_cgcnn_edge(atom_fea_len=atom_fea_len,
                                        nbr_fea_len=nbr_fea_len)])

  layers_list.extend([QEQ(atom_fea_len=atom_fea_len,
                          nbr_fea_len=nbr_fea_len,
                          atom_type=atom_type,
                          activation='relu',
                          e_field=e_field)])
  

  return ListGraphNetwork(layers=layers_list,direct=direct)


class ListGraphNetwork(torch.nn.ModuleList):
    def __init__(self, layers,direct=False):
        """generate a graph network from a sequence of graph modules using build layer function in utils.py
        Args:
            layers (list): prebuilt layers (generated by build_layer)
        """
        self.direct=direct
        super().__init__(layers)

    def forward(self, data):
        if self.direct !=True:
            data['position'].requires_grad=True
        for layer in self:
             data=layer(data)
        return data

