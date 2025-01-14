import torch.nn as nn
from .basic_layers import *
from .basis_layers import *

class RbfEmb(nn.Module):
    
    def __init__ (self,num_radial,nbr_fea_len):
        super(RbfEmb,self).__init__()
        self.rbf_emb=Dense(num_radial,nbr_fea_len,activation=None,bias=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rbf_emb.weight)

    def forward(self,data):
        data['rbf_emb']=self.rbf_emb(data['rbf'])
        return data

class OrbitalEmb(nn.Module):
    
    def __init__(self,lmax,nbr_fea_len):
        super(OrbitalEmb, self).__init__()
        self.lmax=lmax
        self.orbital_emb = Dense(self.lmax**2,nbr_fea_len,activation=None,bias=False)
        
    def forward(self,data):
        data['orbitals_emb']=self.orbital_emb(data['orbitals'])
        return data
        
class AtomEmbeddingLayer(nn.Module):
    # def __init__(self, atom_species, atom_fea_len):
    def __init__(self, atom_type, atom_fea_len,charge=False):
        super(AtomEmbeddingLayer, self).__init__()
        self.atom_type = atom_type
        self.atom_fea_len = atom_fea_len
        self.charge = charge
        self.embedding = Dense(len(self.atom_type), self.atom_fea_len,activation=None,bias=False)
        self.q_embed = Dense(1, self.atom_fea_len, activation=None,bias=False)

    def forward(self, data):
        if self.charge ==True and 'charge' in data.keys():
            data['q_fea'] = self.q_embed(data['charge'].reshape(-1,3,1))
        for i in range(len(self.atom_type)):
            data['atom_number'][data['atom_number']==self.atom_type[i]]=i
        data['atom_attr'] = torch.nn.functional.one_hot(data['atom_number'], len(self.atom_type)).to(device=data['atom_number'].device,dtype=torch.float)
        # if self.charge ==True and 'charge' in data.keys(): #TODO: charge embedding
        #     data['atom_fea'] =  self.q_embed(data['charge'].reshape(-1,1))  #TODO: charge embedding
        # else:  #TODO: charge embedding 
        #     data['atom_fea'] = self.embedding(data['atom_attr'])  #TODO: charge embedding
        data['atom_fea'] = self.embedding(data['atom_attr'])
        return data

class EdgeEmbeddingLayer(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len,num_radial):
        super(EdgeEmbeddingLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_radial = num_radial
        self.edge_input=2*self.atom_fea_len+self.num_radial
        self.embedding= Dense(self.num_radial+2*self.atom_fea_len,self.nbr_fea_len,activation=None,bias=False)
    
    def forward(self,data):
        """
        Args:
            atom_fea: atom feature (N, atom_fea_len)
            rdf: radial distribution function (N*M, num_radial)
            nbr_fea_idx: neighbor feature index (N,M)
        Returns:
            nbr_fea: neighbor feature
        """

        atom_fea=data['atom_fea']
        rbf=data['rbf'] 
        nbr_fea_idx=data['nbr_fea_idx']
        atom_fea=atom_fea[nbr_fea_idx].reshape(-1,self.atom_fea_len*2) #(N_edge, atom_fea_len*2)
        nbr_fea = torch.cat([atom_fea, rbf], dim=-1).to(atom_fea.dtype) #(N*M,num_radial+2*atom_fea_len)
        nbr_fea = self.embedding(nbr_fea) #(N*M,out_fea_len) 
        data['edge']=nbr_fea
        return data

class AEEmbeddingLayer(nn.Module):
    """angle_feature embedding with angle feature

    Args:
        nn (_type_): _description_
    """
    def __init__(self, nbr_fea_len,out_fea_len,num_spherical,num_radial):
        super(AEEmbeddingLayer, self).__init__()
        self.nbr_fea_len = nbr_fea_len
        self.out_fea_len = out_fea_len
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.embedding_layer = Dense(self.num_spherical*self.num_radial+2*self.nbr_fea_len,self.out_fea_len,activation=None,bias=False)
        self.norm=Crystal_Norm(2*self.nbr_fea_len)


    def forward(self,data):
        """_summary_

        Args:
            atom_fea (_type_): _description_
            sbf (_type_): _description_
            angle_fea_idx (_type_): _description_

        Returns:angle_fea
            tensor: # [N_angle,out_fea_len]
        """

        nbr_fea=data['edge']
        sbf=data['sbf']
        angle_nbr_idx=data['angle_nbr_idx']
        crystal_angle_idx=data['crystal_angle_idx']
        K,_ = angle_nbr_idx.shape
        nbr_fea=nbr_fea[angle_nbr_idx].reshape(K,-1) # [K,2,nbr_fea_len] -> [K,2*nbr_fea_len]
        nbr_fea=self.norm(nbr_fea,crystal_angle_idx)
        sbf = sbf.reshape(K,-1) # [K,num_spherical*self.num_radial]
        angle_fea = torch.cat([nbr_fea,sbf],dim=-1).to(self.embedding_layer.weight.dtype) # [K,2*nbr_fea_len+num_spherical*self.num_radial]
        angle_fea = self.embedding_layer(angle_fea) # [K,I,out_fea_len]
        data['angle'] = angle_fea
        return data

class VectorEmbeddinglayer(nn.Module):
    def __init__(self,atom_fea_len,nbr_fea_len,lmax,activation,num_radial):
        super(VectorEmbeddinglayer,self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.activation = activation
        self.lmax = lmax
        self.num_radial=num_radial
        self.vector_embedding = Dense(self.lmax**2+2*self.atom_fea_len+self.num_radial,self.nbr_fea_len,activation=self.activation,bias=False)
    
        self.reset_parameters
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vector_embedding.weight)

    def forward(self,data):
        atom_fea = data['atom_fea']
        rbf=data['rbf'] 
        nbr_fea_idx=data['nbr_fea_idx']
        orbitals=data['orbitals']
        atom_fea=atom_fea[nbr_fea_idx].reshape(-1,self.atom_fea_len*2) #(N_edge, atom_fea_len*2)
        vector_fea = torch.cat([atom_fea, rbf,orbitals], dim=-1).to(self.vector_embedding.weight.dtype) #(N*M,num_radial+2*atom_fea_len)
        vector_emb=self.vector_embedding(vector_fea)
        data['edge']=vector_emb
        return data
