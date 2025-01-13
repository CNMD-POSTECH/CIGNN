from scipy.optimize import brentq
from scipy import special as sp
import sympy as sym
import numpy as np

def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)

def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj

def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')

    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f

def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(x, zeros[order, i]*x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis

def sph_harm_prefactor(l, m):
    """
    Computes the constant pre-factor for the spherical harmonic of degree l and order m
    input:
    l: int, l>=0
    m: int, -l<=m<=l
    """
    return ((2*l+1) * np.math.factorial(l-abs(m)) / (4*np.pi*np.math.factorial(l+abs(m))))**0.5

def associated_legendre_polynomials(l, zero_m_only=True):
    """
    Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
    """
    z = sym.symbols('z')
    P_l_m = [[0]*(j+1) for j in range(l)]

    P_l_m[0][0] = 1
    if l > 0:
        P_l_m[1][0] = z

        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2*j-1)*z*P_l_m[j-1][0] - (j-1)*P_l_m[j-2][0])/j)
        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1-2*i)*P_l_m[i-1][i-1])
                if i + 1 < l:
                    P_l_m[i+1][i] = sym.simplify((2*i+1)*z*P_l_m[i][i])
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        ((2*j-1) * z * P_l_m[j-1][i] - (i+j-1) * P_l_m[j-2][i]) / (j - i))

    return P_l_m

def real_sph_harm(L, spherical_coordinates, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.
    Parameters
    ----------
        L: int
            Degree up to which to calculate the spherical harmonics (degree L is excluded).
        spherical_coordinates: bool
            - True: Expects the input of the formula strings to be phi and theta.
            - False: Expects the input of the formula strings to be x, y and z.
        zero_m_only: bool
            If True only calculate the harmonics where m=0.
    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
            the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L, zero_m_only)
    if zero_m_only:
        # for all m != 0: Y_lm = 0
        Y_l_m = [[0] for l in range(L)]
    else:
        Y_l_m = [[0] * (2 * l + 1) for l in range(L)]  # for order l: -l <= m <= l

    # convert expressions to spherical coordiantes
    if spherical_coordinates:
        # replace z by cos(theta)
        theta = sym.symbols("theta")
        for l in range(L):
            for m in range(len(P_l_m[l])):
                if not isinstance(P_l_m[l][m], int):
                    P_l_m[l][m] = P_l_m[l][m].subs(z, sym.cos(theta))

    ## calculate Y_lm
    # Y_lm = N * P_lm(cos(theta)) * exp(i*m*phi)
    #             { sqrt(2) * (-1)^m * N * P_l|m| * sin(|m|*phi)   if m < 0
    # Y_lm_real = { Y_lm                                           if m = 0
    #             { sqrt(2) * (-1)^m * N * P_lm * cos(m*phi)       if m > 0

    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l, 0) * P_l_m[l][0])  # Y_l0

    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1, L):
            # m > 0
            for m in range(1, l + 1):
                Y_l_m[l][m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, m)
                    * P_l_m[l][m]
                    * sym.cos(m * phi)
                )
            # m < 0
            for m in range(1, l + 1):
                Y_l_m[l][-m] = sym.simplify(
                    2 ** 0.5
                    * (-1) ** m
                    * sph_harm_prefactor(l, -m)
                    * P_l_m[l][m]
                    * sym.sin(m * phi)
                )

        # convert expressions to cartesian coordinates
        if not spherical_coordinates:
            # replace phi by atan2(y,x)
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l in range(L):
                for m in range(len(Y_l_m[l])):
                    Y_l_m[l][m] = sym.simplify(Y_l_m[l][m].subs(phi, sym.atan2(y, x)))
    return Y_l_m