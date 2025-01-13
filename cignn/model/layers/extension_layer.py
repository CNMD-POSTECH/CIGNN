import torch
import numpy as np
import sympy as sym
from typing import Optional

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

# RadialDistributionFunction#### from schnetpack ##########
class CosineCutoff(torch.nn.Module):
    r"""Class of Behler cosine cutoff.
    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    Args:
        cutoff (float, optional): cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """Compute cutoff.
        Args:
            distances (torch.Tensor): values of interatomic distances.
        Returns:
            torch.Tensor: values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs
        
def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.
    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).
    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss

class GaussianSmearing(torch.nn.Module):
    r"""Smear layer using a set of Gaussian functions.
    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.
    """

    def __init__(
        self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = torch.nn.Parameter(widths)
            self.offsets = torch.nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.
        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.
        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.
        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )

class RadialDistribution(torch.nn.Module):
    """
    Radial distribution function used e.g. to compute Behler type radial symmetry functions.
    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        cutoff_function (callable): Cutoff function
    """

    def __init__(self, radial_filter, cutoff_function=CosineCutoff):
        super(RadialDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.cutoff_function = cutoff_function

    def forward(self, r_ij, elemental_weights=None, neighbor_mask=None):
        """
        Args:
            r_ij (torch.Tensor): Interatomic distances
            elemental_weights (torch.Tensor): Element-specific weights for distance functions
            neighbor_mask (torch.Tensor): Mask to identify positions of neighboring atoms
        Returns:
            torch.Tensor: Nbatch x Natoms x Nfilter tensor containing radial distribution functions.
        """

        #nbatch, natoms, nneigh = r_ij.size()
        natoms, nneigh, number = r_ij.size()

        radial_distribution = self.radial_filter(r_ij)

        # If requested, apply cutoff function
        if self.cutoff_function is not None:
            cutoffs = self.cutoff_function(r_ij)
            radial_distribution = radial_distribution * cutoffs.unsqueeze(-1)

        # Apply neighbor mask
        if neighbor_mask is not None:
            radial_distribution = radial_distribution * torch.unsqueeze(
                neighbor_mask, -1
            )

        # Weigh elements if requested
        if elemental_weights is not None:
            radial_distribution = (
                radial_distribution[:, :, :, :, None]
                * elemental_weights[:, :, :, None, :].float()
            )

        radial_distribution = torch.sum(radial_distribution, 2)
        radial_distribution = radial_distribution.view(natoms, nneigh, -1)
        return radial_distribution

#############################################################################
class RBFExpansion(torch.nn.Module):
    """
    from ALIGNN
    Expand interatomic distances with radial basis functions.
    """

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        expand = torch.exp(
            -self.gamma * (distance.unsqueeze(2) - self.centers) ** 2
        )
        return expand.squeeze(2)


class Standardize(torch.nn.Module):
    r"""Standardize layer for shifting and scaling.
    .. math::
       y = \frac{x - \mu}{\sigma}
    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
        eps (float, optional): small offset value to avoid zero division.
    """

    def __init__(self, mean, stddev, eps=1e-9):
        super(Standardize, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)

    def forward(self, input):
        """Compute layer output.
        Args:
            input (torch.Tensor): input data.
        Returns:
            torch.Tensor: layer output.
        """
        # Add small number to catch divide by zero
        y = (input - self.mean) / (self.stddev + self.eps)
        return y


class BehlerAngular(torch.nn.Module):
    def __init__(self, zetas={1}):
        super(BehlerAngular, self).__init__()
        self.zetas = zetas

    def forward(self, cos_theta):
        """
        Args:
            cos_theta (torch.Tensor): Cosines between all pairs of neighbors of the central atom.
        Returns:
            torch.Tensor: Tensor containing values of the angular filters.
        """
        angular_pos = [
            2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_neg = [
            2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_all = angular_pos + angular_neg
        return torch.cat(angular_all, -1)
   
class  AngularDistribution(torch.nn.Module):
    """

    Args:
        torch (_type_): _description_
    """
    def __init__(self,cutoff) :
        super(AngularDistribution,self).__init__()
        self.cutoff = cutoff
        self.angular_filter = BehlerAngular()
        self.cutoff_filter=CosineCutoff(cutoff)
    
    def forward(self,distance1,distance2,angle_degree):
        en=1
        angle_data=self.angular_filter(torch.cos(angle_degree)) 
        data=torch.exp(-en*((distance1)**2+(distance2)**2).unsqueeze(-1))
        f1c=self.cutoff_filter(distance1).unsqueeze(-1)
        f2c=self.cutoff_filter(distance2).unsqueeze(-1)
        angular_dis=angle_data*data*f1c*f2c
        sumed_angular_dis=torch.sum(angular_dis,dim=1)
        return sumed_angular_dis



class GaussianDistance(torch.nn.Module):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step,device='cuda',var=None):
        super(GaussianDistance,self).__init__()
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.device =device
        self.filter = torch.arange(dmin, dmax+step, step).to(self.device)
        #self.filter = torch.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def forward(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        #filter = torch.arange(self.dmin, self.dmax+self.step, self.step).to('cuda')
        #data = torch.exp(-(distances[...,] - self.filter)**2 /self.var**2).to('cuda')
        #return torch.exp(-(distances - self.filter)**2 /self.var**2)
        return  torch.exp(-(distances - self.filter)**2 /self.var**2).to(self.device)