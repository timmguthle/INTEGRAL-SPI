# import ctypes
import math

import numba as nb
import numpy as np
# from typing import Iterable

import astropy.units as astropy_units
# import six
from past.utils import old_div

# import astromodels.functions.numba_functions as nb_func
from astromodels.core.units import get_units
from astromodels.functions.function import (
    Function1D,
    FunctionMeta,
    ModelAssertionViolation,
)

@nb.njit(fastmath=True)#, cache=_cache_functions)
def band_eval(x, K, alpha, beta, E0, piv):

    n = x.shape[0]
    out = np.empty(n)

    break_point = (alpha - beta) * E0

    factor_ab = np.exp(beta - alpha) * math.pow(break_point / piv, alpha - beta)

    for idx in range(n):

        if x[idx] < break_point:
            out[idx] = K * math.pow(x[idx] / piv, alpha) * np.exp(-x[idx] / E0)

        else:

            out[idx] = K * factor_ab * math.pow(x[idx] / piv, beta)

    return out


class C_Band(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        Band model from Band et al., 1993, parametrized with the peak energy, but not really cause i changed it

    latex : $K \begin{cases} \left(\frac{x}{piv}\right)^{\alpha} \exp \left(-\frac{(2+\alpha) x}{x_{p}}\right) & x \leq (\alpha-\beta) \frac{x_{p}}{(\alpha+2)} \\ \left(\frac{x}{piv}\right)^{\beta} \exp (\beta-\alpha)\left[\frac{(\alpha-\beta) x_{p}}{piv(2+\alpha)}\right]^{\alpha-\beta} &x>(\alpha-\beta) \frac{x_{p}}{(\alpha+2)} \end{cases} $

    parameters :

        K :

            desc : Differential flux at the pivot energy
            initial value : 1e-4
            min : 1e-50
            is_normalization : True
            transformation : log10

        alpha :

            desc : low-energy photon index
            initial value : -1.0
            min : -1.5
            max : 3

        xp :

            desc : peak in the x * x * N (nuFnu if x is a energy)
            initial value : 500
            min : 10
            transformation : log10

        beta :

            desc : high-energy photon index
            initial value : -2.0
            min : -5.0
            max : -1.6

        piv :

            desc : pivot energy
            initial value : 100.0
            fix : yes
    """

    def _set_units(self, x_unit, y_unit):
        # The normalization has the same units as y
        self.K.unit = y_unit

        # The break point has always the same dimension as the x variable
        self.xp.unit = x_unit

        self.piv.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, xp, beta, piv):
        E0 = xp

        if alpha < beta:
            raise ModelAssertionViolation("Alpha cannot be less than beta")

        if isinstance(x, astropy_units.Quantity):
            alpha_ = alpha.value
            beta_ = beta.value
            K_ = K.value
            E0_ = E0.value
            piv_ = piv.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            alpha_, beta_, K_, piv_, x_, E0_ = alpha, beta, K, piv, x, E0

        return band_eval(x_, K_, alpha_, beta_, E0_, piv_) * unit_
    

@nb.njit(fastmath=True)
def beuermann_eval(x, K, alpha, beta, n, E1, E2):
    return K*np.power(np.power(x/E1,-alpha*n)+np.power(x/E2,-beta*n),-1/n)

class Beuermann(Function1D, metaclass=FunctionMeta):
    r"""
    description :

        Beuermann function. similar to the band function but with a parameter that controls the smoothness of the transition between the two power laws independently of the slopes.

    latex : K \left( \left( \frac{x}{E_{\mathrm{1}}} \right) ^ {- \alpha n} + \left( \frac{x}{E_{\mathrm{2}}} \right) ^ {- \beta n} \right) ^ {-\frac{1}{n}}

    parameters :

        K :
            
            desc : Normalization of the complete function
            initial value : 0.1
            min : 1e-50
            is_normalization : True
            transformation : log10

        alpha :

            desc : low-energy photon index
            initial value : -1.0
            min : -4.0
            max : 3

        beta :

            desc : high-energy photon index
            initial value : -2.0
            min : -5.0
            max : -1.6

        n :
            desc : smoothness parameter
            initial value : 1
            min : 0
            max : 100

        E1 :

            desc : energy for low energy power law, for break point
            initial value : 100.0


        E2 :
            
            desc : energy for high energy power law for break point
            initial value : 100.0

    """

    def _set_units(self, x_unit, y_unit):
        # The normalization has the same units as y
        self.K.unit = y_unit
        # The break point has always the same dimension as the x variable
        # self.xp.unit = x_unit

        self.E1.unit = x_unit
        self.E2.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled
        self.n.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, beta, n, E1, E2):

        # if alpha < beta:
        #     raise ModelAssertionViolation("Alpha cannot be less than beta")

       
        if isinstance(x, astropy_units.Quantity):
            alpha_ = alpha.value
            beta_ = beta.value
            n_ = n.value
            K = K.value
            E1_ = E1.value
            E2_ = E2.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            x_, alpha_, beta_, n_, K_, E1_, E2_ = x, alpha, beta, n, K, E1, E2

        return beuermann_eval(x_, K_, alpha_, beta_, n_, E1_, E2_) * unit_
    


