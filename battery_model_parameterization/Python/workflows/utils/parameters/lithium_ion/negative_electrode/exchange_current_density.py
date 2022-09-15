#
# Modifies pybamm exchange current density functions to allow j0_n
# and/ or alpha_n as inputs.
#
import pybamm
from functools import partial


def _graphite_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, T, alpha_input, j0_input
):
    """
    Modifies pybamm function to allow j0_n and alpha_n input.
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.
    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    Parameters
    ----------
    c_e: pybamm.Symbol
        Electrolyte concentration [mol.m-3]
    c_s_surf:pybamm.Symbol
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
    m_ref, alpha = 2 * 10 ** (-5), 0.5
    if alpha_input:
        alpha = pybamm.InputParameter("alpha_n")
    if j0_input:
        m_ref = pybamm.InputParameter("j0_n")
    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_n_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_n_max - c_s_surf) ** alpha
    )


def graphite_electrolyte_exchange_current_density_Dualfoil1998(alpha_input, j0_input):
    return partial(
        _graphite_electrolyte_exchange_current_density_Dualfoil1998,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )


def _graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, T, alpha_input, j0_input
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
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
    m_ref, alpha = 6.48e-7, 0.5
    if j0_input:
        m_ref = pybamm.InputParameter(
            "j0_n"
        )  # 6.48e-7 (A/m2)(mol/m3)**1.5 - includes ref concentrations
    if alpha_input:
        alpha = pybamm.InputParameter("alpha_n")
    E_r = 35000
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_n_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_n_max - c_s_surf) ** alpha
    )


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(alpha_input, j0_input):
    return partial(
        graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )


def _graphite_electrolyte_exchange_current_density_PeymanMPM(
    c_e, c_s_surf, c_s_max, T, alpha_input, j0_input
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.
    Check the unit of Reaction rate constant k0 is from Peyman MPM.
    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
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
    m_ref, alpha = 1.061 * 10 ** (-6), 0.5
    # m_ref units are (A/m2)(mol/m3)**1.5 - includes ref concentrations

    if j0_input:
        m_ref = pybamm.InputParameter("j0_n")
    if alpha_input:
        alpha = pybamm.InputParameter("alpha_n")

    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_s_max - c_s_surf) ** alpha
    )


def graphite_electrolyte_exchange_current_density_PeymanMPM(alpha_input, j0_input):
    return partial(
        _graphite_electrolyte_exchange_current_density_PeymanMPM,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )
