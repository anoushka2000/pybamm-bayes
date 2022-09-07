#
# Modifies pybamm exchange current density functions to allow Ds_n as an input.
#
import pybamm


def graphite_mcmb2528_diffusivity_Dualfoil1998(sto, T):
    """
    Graphite MCMB 2528 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].
    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = pybamm.InputParameter("Ds_n")  # default = 3.9 * 10 ** (-14)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
