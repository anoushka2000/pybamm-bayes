#
# Modifies pybamm exchange current density functions to allow j0_p as an input.
#

import pybamm


def lico2_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, T):
    """
    Exchange-current density for Butler-Volmer reactions between lico2 and LiPF6 in
    EC:DMC.
    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = pybamm.InputParameter(
        "j0_p"
    )  # default = 6 * 10 ** (-7) (A/m2)(mol/m3)**1.5 - includes ref concentrations
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


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, T):
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
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = pybamm.InputParameter(
        "j0_p"
    )  # default value: m_ref = 3.42e-6 (A/m2)(mol/m3)**1.5 - includes ref concentrations

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


def NMC_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.
    References
    ----------
    .. Peyman MPM manuscript (to be submitted)
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = pybamm.InputParameter("j0_p")
    # default: m_ref = 4.824 * 10 ** (-6)   (A/m2)(mol/m3)**1.5 - includes ref concentrations
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
