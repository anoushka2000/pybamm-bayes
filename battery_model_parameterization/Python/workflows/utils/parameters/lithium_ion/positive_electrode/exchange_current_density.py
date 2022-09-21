#
# Modifies pybamm exchange current density functions to allow j0_p
# and/or alpha_p as inputs.
#

from functools import partial

import pybamm


def _lico2_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, T, alpha_input, j0_input
):
    """
    Exchange-current density for Butler-Volmer reactions between lico2 and LiPF6 in
    EC:DMC.
    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    Parameters
    ----------
    c_e: pybamm.Symbol
        Electrolyte concentration [mol.m-3]
    c_s_surf: pybamm.Symbol
        Particle concentration [mol.m-3]
    T: pybamm.Symbol
        Temperature [K]
    alpha_input: bool
        If True set alpha (Butler-Volmer charge tranfer coefficient)
        as an input.
    j0_input: bool
        If True set j0 (reference exchange current density)
        as an input.
    Returns
    -------
    pybamm.Symbol
        Exchange-current density [A.m-2]
    """
    m_ref, alpha = 6 * 10 ** (-7), 0.5

    if j0_input:
        m_ref = pybamm.InputParameter("j0_p")
    if alpha_input:
        alpha = pybamm.InputParameter("alpha_p")  # default = 0.5
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_p_max - c_s_surf) ** alpha
    )


def lico2_electrolyte_exchange_current_density_Dualfoil1998(alpha_input, j0_input):
    return partial(
        _lico2_electrolyte_exchange_current_density_Dualfoil1998,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )


def _nmc_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, T, alpha_input, j0_input
):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.
    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.
    Parameters
    ----------
    c_e: pybamm.Symbol
        Electrolyte concentration [mol.m-3]
    c_s_surf: pybamm.Symbol
        Particle concentration [mol.m-3]
    T: pybamm.Symbol
        Temperature [K]
    alpha_input: bool
        If True set alpha (Butler-Volmer charge tranfer coefficient)
        as an input.
    j0_input: bool
        If True set j0 (reference exchange current density)
        as an input.
    Returns
    -------
    pybamm.Symbol
        Exchange-current density [A.m-2]
    """
    m_ref, alpha = 3.42e-6, 0.5
    # includes ref concentrations

    if j0_input:
        m_ref = pybamm.InputParameter("j0_p")
    if alpha_input:
        alpha = pybamm.InputParameter("alpha_p")

    E_r = 17800
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_p_max - c_s_surf) ** alpha
    )


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(alpha_input, j0_input):
    return partial(
        _nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )


def _NMC_electrolyte_exchange_current_density_PeymanMPM(
    c_e, c_s_surf, c_s_max, T, alpha_input, j0_input
):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.
    References
    ----------
    .. Peyman MPM manuscript (to be submitted)
    Parameters
    ----------
    c_e: pybamm.Symbol
        Electrolyte concentration [mol.m-3]
    c_s_surf: pybamm.Symbol
        Particle concentration [mol.m-3]
    c_s_max: pybamm.Symbol
        Maximum particle concentration [mol.m-3]
    T: pybamm.Symbol
        Temperature [K]
    alpha_input: bool
        If True set alpha (Butler-Volmer charge tranfer coefficient)
        as an input.
    j0_input: bool
        If True set j0 (reference exchange current density)
        as an input.

    Returns
    -------
    pybamm.Symbol
        Exchange-current density [A.m-2]
    """
    m_ref, alpha = 4.824 * 10 ** (-6), 0.5

    # m_ref (A/m2)(mol/m3)**1.5 - includes ref concentrations
    if j0_input:
        m_ref = pybamm.InputParameter("j0_p")
    if alpha_input:
        alpha = pybamm.InputParameter("alpha_p")
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_s_max - c_s_surf) ** alpha
    )


def NMC_electrolyte_exchange_current_density_PeymanMPM(alpha_input, j0_input):
    return partial(
        _NMC_electrolyte_exchange_current_density_PeymanMPM,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )
