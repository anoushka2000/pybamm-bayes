import pybamm

import battery_model_parameterization as bmp


def _exchange_current_density_inputs(variable_names):
    j0_n_input = "j0_n" in variable_names
    j0_p_input = "j0_p" in variable_names
    alpha_n_input = "alpha_n" in variable_names
    alpha_p_input = "alpha_p" in variable_names
    return j0_n_input, j0_p_input, alpha_n_input, alpha_p_input


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
        ] = bmp.graphite_mcmb2528_diffusivity_Dualfoil1998

    if "Ds_p" in variable_names:
        param[
            "Positive electrode diffusivity [m2.s-1]"
        ] = bmp.lico2_diffusivity_Dualfoil1998

    if "De" in variable_names:
        param[
            "Electrolyte diffusivity [m2.s-1]"
        ] = bmp.electrolyte_diffusivity_Capiglia1999

    (
        j0_n_input,
        j0_p_input,
        alpha_n_input,
        alpha_p_input,
    ) = _exchange_current_density_inputs(variable_names)

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = bmp.graphite_electrolyte_exchange_current_density_Dualfoil1998(
        alpha_input=alpha_n_input, j0_input=j0_n_input
    )

    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = bmp.lico2_electrolyte_exchange_current_density_Dualfoil1998(
        alpha_input=alpha_p_input, j0_input=j0_p_input
    )

    return param


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
        param["Negative electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_n")

    if "Ds_p" in variable_names:
        param["Positive electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_p")

    if "De" in variable_names:
        param[
            "Electrolyte diffusivity [m2.s-1]"
        ] = bmp.electrolyte_diffusivity_Nyman2008

    (
        j0_n_input,
        j0_p_input,
        alpha_n_input,
        alpha_p_input,
    ) = _exchange_current_density_inputs(variable_names)

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = bmp.graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
        alpha_input=alpha_n_input, j0_input=j0_n_input
    )

    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = bmp.nmc_LGM50_electrolyte_exchange_current_density_Chen2020(
        alpha_input=alpha_p_input, j0_input=j0_p_input
    )

    return param


def mohtat_2020(variables):
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
    pybamm.ParameterValues object with values from Mohtat 2020.
    """

    param = pybamm.ParameterValues("Chen2020")
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
        ] = bmp.graphite_diffusivity_PeymanMPM

    if "Ds_p" in variable_names:
        param["Positive electrode diffusivity [m2.s-1]"] = bmp.NMC_diffusivity_PeymanMPM

    if "De" in variable_names:
        param[
            "Electrolyte diffusivity [m2.s-1]"
        ] = bmp.electrolyte_diffusivity_PeymanMPM

    (
        j0_n_input,
        j0_p_input,
        alpha_n_input,
        alpha_p_input,
    ) = _exchange_current_density_inputs(variable_names)

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = bmp.graphite_electrolyte_exchange_current_density_PeymanMPM(
        alpha_input=alpha_n_input, j0_input=j0_n_input
    )

    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = bmp.NMC_electrolyte_exchange_current_density_PeymanMPM(
        alpha_input=alpha_n_input, j0_input=j0_n_input
    )

    return param
