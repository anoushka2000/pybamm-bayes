import pybamm


def lico2_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, T):
    """
    Modifies pybamm function to allow j0_p input.
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
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_p_max - c_s_surf) ** 0.5
    )


def graphite_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, T):
    """
    Modifies pybamm function to allow j0_n input.
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
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
        "j0_n"
    )  # default = (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_n_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_n_max - c_s_surf) ** 0.5
    )


def dfn_constant_current_discharge(d_rate):
    """
    DFN model with diffusivities and reference exchange current densities as inputs.

    Parameters
    ----------
    d_rate : float
        Discharge rate.

    Returns
    -------
    model: pybamm.models.full_battery_models.lithium_ion.dfn.DFN
    param: pybamm.parameters.parameter_values.ParameterValues
    """
    model = pybamm.lithium_ion.DFN()

    param = model.default_parameter_values
    param["Negative electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_n")
    param["Positive electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_p")
    param["Electrolyte diffusivity [m2.s-1]"] = pybamm.InputParameter("De")
    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = graphite_electrolyte_exchange_current_density_Dualfoil1998
    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = lico2_electrolyte_exchange_current_density_Dualfoil1998
    param["Current function [A]"] = param["Nominal cell capacity [A.h]"] * d_rate
    return model, param
