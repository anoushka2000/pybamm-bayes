from functools import partial

import pybamm
from pybamm_bayes.workflows.utils.parameter_sets.utils import (
    _exchange_current_density_inputs,
)


def electrolyte_diffusivity_Capiglia1999(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].
    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
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

    D_c_e = pybamm.InputParameter("De") * pybamm.exp(
        -0.65 * c_e / 1000
    )  # default = 5.34e-10
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


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


def _graphite_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, c_s_max, T, alpha_input, j0_input
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


def lico2_diffusivity_Dualfoil1998(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
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
    D_ref = pybamm.InputParameter("Ds_p")  # default = 1 * 10 ** (-13)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def _lico2_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, c_s_max, T, alpha_input, j0_input
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


def marquis_2019(variables):
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
    pybamm.ParameterValues object with values from Marquis 2019.
    """

    param = pybamm.ParameterValues("Marquis2019")
    variable_names = [v.name for v in variables]

    if "am_fraction_p" in variable_names:
        param[
            "Positive electrode active material volume fraction"
        ] = pybamm.InputParameter("am_fraction_p")

    if "am_fraction_n" in variable_names:
        param[
            "Negative electrode active material volume fraction"
        ] = pybamm.InputParameter("am_fraction_n")

    if "t_+" in variable_names:
        param["Cation transference number"] = pybamm.InputParameter("t_+")

    if "Ds_n" in variable_names:
        param[
            "Negative electrode diffusivity [m2.s-1]"
        ] = graphite_mcmb2528_diffusivity_Dualfoil1998

    if "Ds_p" in variable_names:
        param[
            "Positive electrode diffusivity [m2.s-1]"
        ] = lico2_diffusivity_Dualfoil1998

    if "De" in variable_names:
        param["Electrolyte diffusivity [m2.s-1]"] = electrolyte_diffusivity_Capiglia1999

    (
        j0_n_input,
        j0_p_input,
        alpha_n_input,
        alpha_p_input,
    ) = _exchange_current_density_inputs(variable_names)

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = graphite_electrolyte_exchange_current_density_Dualfoil1998(
        alpha_input=alpha_n_input, j0_input=j0_n_input
    )

    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = lico2_electrolyte_exchange_current_density_Dualfoil1998(
        alpha_input=alpha_p_input, j0_input=j0_p_input
    )

    return param
