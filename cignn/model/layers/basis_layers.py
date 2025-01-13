import time
import math
import torch
import numpy as np
import sympy as sym
import torch.nn as nn
from cignn.utils.base_util import bessel_basis, real_sph_harm

class Envelope(torch.nn.Module):
    """
    Envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        p: int
            Exponent of the envelope function.
    """

    def __init__(self, p, name="envelope"):
        super().__init__()
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled)) 

class BesselBasisLayer(torch.nn.Module):
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        name="bessel_basis",
    ):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5

        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.Tensor(
                np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32)
            ),
            requires_grad=True,
        )

    def forward(self, data): # (natom,nedge,1)
        d = data['distance']
        d_scaled = d.reshape(-1,1) * self.inv_cutoff #(natom*nedge,1)
        env = self.envelope(d_scaled)
        data['rbf']= env * self.norm_const * torch.sin(self.frequencies * d_scaled) / d.reshape(-1,1) #(batch atom* nEdges, nRadial)
        return data
        
class SphericalBasisLayer(torch.nn.Module):
    """
    2D Fourier Bessel Basis
    지금 있는것은 a-b-c가 있을때 a-b에 대한 edge에 대하여 abc angle의 sph를 구하는것임
    
    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient: bool = False,
        name: str = "spherical_basis",
    ):
        super().__init__()

        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = 1 / cutoff

        # retrieve formulas
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        self.sph_funcs = []  # (num_spherical,)
        self.bessel_funcs = []  # (num_spherical * num_radial,)
        self.norm_const = self.inv_cutoff ** 1.5
        self.register_buffer(
            "device_buffer", torch.zeros(0), persistent=False
        )  # dummy buffer to get device of layer

        # convert to torch functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0  # only single angle
        for l in range(len(Y_lm)):  # num_spherical
            if l == 0: 
                # Y_00 is only a constant -> function returns value and not tensor
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: torch.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )
    def ragged_range(self,sizes):
        """
        -------
        Example
        -------
            sizes = [1,3,2] ;
            Return: [0  0 1 2  0 1]
        """
        a = torch.arange(sizes.max())
        indices = torch.empty(sizes.sum(), dtype=torch.long)
        start = 0
        for size in sizes:
            end = start + size
            indices[start:end] = a[:size]
            start = end
        return indices

    def forward(self, data):
        """_summary_

        Args:
            D_ca (tensor): distance,shape(natom,nedge,1)
            Angle_cab (tensor): angle for radian shape(natom,angle,1)
            angle_nbr_idx (tensor): index of main edgem,shape(natom,angle,2)
            

        Returns:
            _type_: _description_
        """
        distance=data['distance']
        angle=data['angle_radi']
        bond_idx=data['angle_nbr_idx']
        D_ca = distance.reshape(-1) # (natom*nedge)
        Angle_cab = angle.reshape(-1) # (natom*angle)
        id3_reduce_ca = bond_idx[:,0].reshape(-1) # (natom*angle)
        #_, K = np.unique(id3_reduce_ca, return_counts=True)
        #Kidx3=self.ragged_range(torch.tensor(K)) # (natom*angle)
        d_scaled = D_ca * self.inv_cutoff  # (nEdges,)
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        # num_spherical: 0 0 0 0 1 1 1 1 ...
        # num_radial: 0 1 2 3 0 1 2 3 ...
        rbf = torch.stack(rbf, dim=1)  # (natom*nEdges, num_spherical * num_radial)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf  # (natom*nEdges, num_spherical * num_radial)
        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = torch.stack(sph, dim=1)  # (nTriplets, num_spherical)
        # if not self.efficient:
        rbf_env = rbf_env[id3_reduce_ca]  # (nTriplets, num_spherical * num_radial)
        rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
        #print(rbf_env.shape)
        # e.g. num_spherical = 3, num_radial = 2
        # z_ln: l: 0 0  1 1  2 2
        #       n: 0 1  0 1  0 1
        sph = sph.view(-1, self.num_spherical, 1)  # (nTriplets, num_spherical, 1)
        #print(sph.shape)
        # e.g. num_spherical = 3, num_radial = 2
        # Y_lm: l: 0 0  1 1  2 2
        #       m: 0 0  0 0  0 0
        out = (rbf_env * sph).view(-1, self.num_spherical * self.num_radial)
   
        data['sbf']=out  # (nTriplets, num_spherical * num_radial)
        return data