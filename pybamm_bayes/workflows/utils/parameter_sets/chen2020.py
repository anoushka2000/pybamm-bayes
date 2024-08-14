from functools import partial

import pybamm

from pybamm_bayes.workflows.utils.parameter_sets.utils import (
    _exchange_current_density_inputs,
)


def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]
    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.
    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = (
        pybamm.InputParameter("De") * (c_e / 1000) ** 2
        - 3.972e-10 * (c_e / 1000)
        + 4.862e-10
    )
    # default = 8.794e-11
    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def _graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T, alpha_input, j0_input
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
        m_ref * arrhenius * c_e**alpha * c_s_surf**alpha * (c_n_max - c_s_surf) ** alpha
    )


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(alpha_input, j0_input):
    return partial(
        _graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )


def _nmc_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T, alpha_input, j0_input
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
        m_ref * arrhenius * c_e**alpha * c_s_surf**alpha * (c_p_max - c_s_surf) ** alpha
    )


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(alpha_input, j0_input):
    return partial(
        _nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
        alpha_input=alpha_input,
        j0_input=j0_input,
    )


def chen_2020(variables):
    """
    Creates a parameter set with variables as inputs.

    Parameters
    __________
    variables: List[Variables]
        List of variable objects.
        Variable names should follow:
        - Electrode active material volume fractions: am_fraction_n, am_fraction_p
        - Electrode Diffusivities: Ds_n, Ds_p, De
        - Electrode Diffusivity: De
        - Electrode reference exchange-current densities: j0_n, j0_p
        - Cation transference number = t_+

    Returns
    -------
    pybamm.ParameterValues object with values from Chen 2020.
    """

    param = pybamm.ParameterValues("Chen2020")
    variable_names = [v.name for v in variables]

    if "am_fraction_p" in variable_names:
        param["Positive electrode active material volume fraction"] = (
            pybamm.InputParameter("am_fraction_p")
        )

    if "am_fraction_n" in variable_names:
        param["Negative electrode active material volume fraction"] = (
            pybamm.InputParameter("am_fraction_n")
        )

    if "t_+" in variable_names:
        param["Cation transference number"] = pybamm.InputParameter("t_+")

    if "Ds_n" in variable_names:
        param["Negative electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_n")

    if "Ds_p" in variable_names:
        param["Positive electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_p")

    if "De" in variable_names:
        param["Electrolyte diffusivity [m2.s-1]"] = electrolyte_diffusivity_Nyman2008

    (
        j0_n_input,
        j0_p_input,
        alpha_n_input,
        alpha_p_input,
    ) = _exchange_current_density_inputs(variable_names)

    param["Negative electrode exchange-current density [A.m-2]"] = (
        graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
            alpha_input=alpha_n_input, j0_input=j0_n_input
        )
    )

    param["Positive electrode exchange-current density [A.m-2]"] = (
        nmc_LGM50_electrolyte_exchange_current_density_Chen2020(
            alpha_input=alpha_p_input, j0_input=j0_p_input
        )
    )

    return param
