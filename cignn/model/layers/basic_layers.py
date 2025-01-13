import torch
import torch.nn as nn
from torch_scatter import scatter

class Dense(nn.Module):
    """
    Combine dense layers with activation functions.
    Parameters
    ----------
        in_features: int
            input embedding size.
        out_features: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        
        if isinstance(activation, str):
            activation = activation.lower()
        if activation == 'softplus':
            self._activation = nn.Softplus()
        elif activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'silu':
            self._activation = nn.SiLU()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'gelu':
            self._activation = nn.GELU()
        elif activation is None:
            self._activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GNN (yet)."
            )
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight)
        if self.linear.bias is not None:
           self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = x.to(self.linear.weight.dtype)
        x = self.linear(x)
        x = self._activation(x)
        return x
    
class ResidualLayers(nn.Module):
    """
    residual layer from Gemnet
    """
    def __init__ (self, n_layers, n_in, n_mid, activation='relu'):
        super().__init__()
        assert n_layers > 1, "n_layer must be greater than 1"
        self.n_layers = n_layers
        self.n_in = n_in
        self.n_mid = n_mid
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(Dense(n_in, n_mid, activation=None))
            elif i == int(n_layers) - 1:
                self.layers.append(Dense(n_mid, n_in, activation=None))
            else:
                self.layers.append(Dense(n_mid, n_mid, activation=None))
        self.inv_sqrt_2 = 1 / (2.0) ** 0.5
        self.acts=nn.ModuleList([nn.SiLU() for _ in range(n_layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(n_mid) if i!=0 else nn.LayerNorm(n_in) for i in range(n_layers)])
        
    def forward(self, input):
        x = input
        for i in range(self.n_layers):
            x = self.norms[i](x)
            x = self.acts[i](x)
            x = self.layers[i](x)
        x = input + x
        x = x * self.inv_sqrt_2
        x_mean =x.mean() #TODO remove
        x_std = x.std() # TODO remove

        return x

class OutData(nn.Module):
    """
    precidct energy from atom_fea and force from edge_fea
    Args:
        atom_fea: atom feature
        edge_fea: edge feature
    Returns:
        energy: energy of the system
        force: force of the system
    """
    def __init__(self, atom_fea_len, nbr_fea_len,h_fea_len,n_h,num_radial, activation,direct=False):
        super().__init__()
        self.n_h =n_h
        self.atom_fea_len = atom_fea_len
        self.edge_fea_len = nbr_fea_len
        self.activation = activation
        self.reduce_fea_node = [self.atom_fea_len//2**x for x in range(n_h+1)]
        self.energy_fc = nn.ModuleList([Dense(self.reduce_fea_node[x],self.reduce_fea_node[x+1],activation=activation,bias=None) for x in range(self.n_h)])
        self.out_energy = Dense(self.reduce_fea_node[-1], 1, activation=None)
        self.reduce_fea_edge = [self.edge_fea_len//2**x for x in range(n_h+1)]
        self.force_fc = nn.ModuleList([Dense(self.reduce_fea_edge[x],self.reduce_fea_edge[x+1],activation=activation,bias=None) for x in range(self.n_h)])
        self.out_force =  Dense(self.reduce_fea_edge[-1], 1, activation=None,bias=None)
        self.mlp_rbf = Dense(num_radial,nbr_fea_len,activation=None,bias=None)
        self.direct = direct
        self.layer_norm=nn.LayerNorm(atom_fea_len)


    def forward(self,data):
        atom_fea=data['atom_fea']
        atomic_energy=atom_fea
        for _,efc in enumerate(self.energy_fc):
            atomic_energy = efc(atomic_energy)
        atomic_energy = self.out_energy(atomic_energy) #(n_atom,1)
        
        if self.direct :
            nbr_fea_idx=data['nbr_fea_idx']
            nbr_swap_idx=data['nbr_swap_idx']
            edge_fea=data['edge']
            rbf=data['rbf']
            rbf_f=self.mlp_rbf(rbf)
            atom_nbr=atom_fea[nbr_fea_idx]
            atom_sim= torch.matmul(atom_nbr[:,0,:].unsqueeze(1),torch.transpose(atom_nbr[:,1,:].unsqueeze(1),2,1)).squeeze(1)/(torch.norm(atom_nbr[:,0,:].unsqueeze(1),dim=2)*torch.norm(atom_nbr[:,1,:].unsqueeze(1),dim=2))
            edge_fea = edge_fea * rbf_f 
            for _, ffc in enumerate(self.force_fc):
                edge_fea = ffc(edge_fea)
            force = self.out_force(edge_fea) #(n_edge,1)
            force = -1*force*atom_sim
            force_swap = force[nbr_swap_idx] #(n_edge,1)
            force =torch.mean(torch.cat((force.unsqueeze(1),force_swap.unsqueeze(1)),dim=1),dim=1) #(n_edge,1)
            if 'atomic_energy' in data.keys():
                data['atomic_energy'] = data['atomic_energy'] + atomic_energy
            else:
                data['atomic_energy'] = atomic_energy
                
            if 'force' in data.keys():
                data['force'] = data['force'] + force
            else:
                data['force'] = force
            return data
        else:
            if 'atomic_energy' in data.keys():
                data['atomic_energy'] = data['atomic_energy'] + atomic_energy
            else:
                data['atomic_energy'] = atomic_energy
            return data

class ElementScale(torch.nn.Module):
    def __init__(
            self, 
            atom_type,
            eps=1e-6,
            momentum=0.1,
            unbiased=True,
            Trainable=False,
            type='species', # species or system
            systemE:torch.Tensor=None,
            QeqE:torch.Tensor=None,
            scale:dict=None,
            divider:torch.Tensor=None):
        super().__init__()
        self.divider = divider
        self.atom_type = atom_type
        
        if type == 'species':
            systemE=torch.tensor(list(systemE.values()),requires_grad=Trainable)
            if QeqE != None:
                QeqE=torch.tensor(list(QeqE.values()),requires_grad=Trainable)
                real_shift=systemE-QeqE
            else:
                real_shift=systemE
            systemE=real_shift/divider
            
        elif type == 'system':
            if QeqE != None:
                real_shift=systemE-QeqE
            else:
                real_shift=systemE
            real_shift=systemE-QeqE
            systemE = (real_shift.repeat(len(scale)))/divider
        else:
            raise ValueError('type must be species or system')
        
        scale=torch.tensor(list(scale.values()),requires_grad=Trainable)
        scale = scale/divider
     

        if scale != None:
            self.scale =torch.nn.Parameter(scale,requires_grad=Trainable).to(self.atom_type.device)
        else:
            self.scale = torch.nn.Parameter(torch.ones(len(self.atom_type),requires_grad=Trainable))
    
        if systemE != None:
            self.shift =torch.nn.Parameter(systemE,requires_grad=Trainable).to(self.atom_type.device)
            
        else:
            self.shift = torch.nn.Parameter(torch.zeros(len(self.atom_type),requires_grad=Trainable))
            
        self.momentum = momentum
        self.eps = eps
        self.unbiased=unbiased

    def forward(self,data):
        atom_numbers=data['atom_number']
        atomic_energy=data['atomic_energy']

        if len(atomic_energy.shape) != 1:
            atomic_energy = atomic_energy.squeeze()  
        data['element_shift']=self.shift
        data['element_scale']=self.scale

        data['atomic_energy']=(atomic_energy*self.scale[atom_numbers]+self.shift[atom_numbers])
        return data

class Fin_Out(torch.nn.Module):
    def __init__(self,direct,divider=1.0):
        super(Fin_Out,self).__init__()
        self.direct = direct
        self.divider = divider
    def forward(self,data):
        energy=scatter(data['atomic_energy'],data['crystal_atom_idx'],dim=0,reduce='add').reshape(-1,1)
        # if 'static_e' in data.keys():
        #      energy= energy + data['static_e'].reshape(-1,1)

        if self.direct:
            force_swap = data['force'][data['nbr_swap_idx']] # (n_edge,1)
            force = torch.mean(torch.cat([data['force'].unsqueeze(1),force_swap.unsqueeze(1)],dim=1),dim=1) # (n_edge,1)
            force = force * (data['nbr_vec']/(data['distance'].unsqueeze(1))) #(n_edge, 3)
            force = scatter(force, data['nbr_fea_idx'][:,0], dim=0, reduce='add') # (num_atom,3)
            force = force/self.divider
        else:
            force = -torch.autograd.grad(energy.sum(), data['position'],retain_graph=self.training,create_graph=self.training)[0] # (num_atom,3)
       
        atom_number=torch.unique(data['crystal_atom_idx'],return_counts=True)[1].reshape(-1,1)
        data['total_energy']=energy 
        data['energy']= data['total_energy']/atom_number
        data['force']=force
        return data

class Crystal_Norm(nn.Module):
    def __init__(self, norm_fea_len,unbiased=True, eps=1e-6, momentum=0.1):
        super(Crystal_Norm,self).__init__()
        self.norm_fea_len = norm_fea_len
        self.weight = nn.Parameter(torch.empty((self.norm_fea_len), requires_grad=True))
        self.bias = nn.Parameter(torch.empty((self.norm_fea_len), requires_grad=True))
        self.eps = eps
        self.momentum = momentum
        self.unbiased=unbiased
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


    def forward(self, target_fea, index):  
        device = target_fea.device
        # Check the shape of inputs
        if target_fea.shape[0] != index.shape[0]:
            raise ValueError('Expected target_fea and index to have the same size in the 0th dimension.')

        mean = scatter(target_fea, index, dim=0, reduce='mean')

        if index.max() + 1 < mean.shape[0]:
            raise ValueError('Expected max value of index to be no less than the 0th dimension of mean.')
        
        # Compute variance and standard deviation
        if  self.unbiased == False:
            var = scatter((target_fea - mean[index]) ** 2, index, dim=0, reduce='mean')
            std = torch.sqrt(var.clamp_min(1e-7))
        else:
            squared_diffs=(target_fea-mean[index])**2
            diffs_sum=scatter(squared_diffs,index,dim=0,reduce='sum') + self.eps
            nums=torch.unique(index,return_counts=True)[1].reshape(-1,1)
            if diffs_sum.shape[0] != nums.shape[0]:
                # unique_data, counts=torch.unique(index,return_counts=True)
                unique_data,count=torch.unique(index,return_counts=True)
                all_values = torch.arange(0, unique_data.max() + 1,device=device)
                mask = (all_values.unsqueeze(1) == unique_data.unsqueeze(0)).any(dim=1)
                missing_indices = torch.nonzero(~mask).squeeze()
                missing_indices = missing_indices.reshape(-1).to(device)
                max_number=torch.max(torch.concat((unique_data,missing_indices),dim=0))
                count_data=torch.zeros(max_number+1,device=device)
                count_data[unique_data]=count.float()

                nums = count_data.reshape(-1,1)

            var =diffs_sum/(nums -1)

            std = torch.sqrt(var.clamp_min(1e-7))

        target_norm = (target_fea - mean[index]) / (std[index] + self.eps)
        target_norm = target_norm * self.weight + self.bias
        return target_norm

class SoftMaxCustom(nn.Module):

    def __init__(self,dim, eps=1e-5):
        super(SoftMaxCustom,self).__init__()
        self.dim=dim
        self.eps=eps

    def forward(self,data,index):
        # data=torch.exp(data)+self.eps
        max_data=scatter(data,index,dim=self.dim,reduce='min')
        data=torch.exp(data-max_data[index])+self.eps
        under=scatter(data , index , dim=self.dim,reduce='sum')
        output=data/(under[index])
        return output