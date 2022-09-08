import pybamm
import battery_model_parameterization as bmp


def marquis_2019():
    """
    Returns
    -------
    pybamm.ParameterValues object with values from Marquis 2019 and inputs:
    - Electrode active material volume fractions: am_fraction_n, am_fraction_p
    - Electrode Diffusivities: Ds_n, Ds_p, De
    - Electrode Diffusivity: De
    - Electrode reference exchange-current densities: j0_n, j0_p
    - Cation transference number = t_+
    """

    param = pybamm.ParameterValues("Marquis2019")

    param["Positive electrode active material volume fraction"] = pybamm.InputParameter(
        "am_fraction_p"
    )
    param["Negative electrode active material volume fraction"] = pybamm.InputParameter(
        "am_fraction_n"
    )

    param["Cation transference number"] = pybamm.InputParameter("t_+")

    param[
        "Negative electrode diffusivity [m2.s-1]"
    ] = bmp.graphite_mcmb2528_diffusivity_Dualfoil1998
    param[
        "Positive electrode diffusivity [m2.s-1]"
    ] = bmp.lico2_diffusivity_Dualfoil1998
    param["Electrolyte diffusivity [m2.s-1]"] = bmp.electrolyte_diffusivity_Capiglia1999

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = bmp.graphite_electrolyte_exchange_current_density_Dualfoil1998
    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = bmp.lico2_electrolyte_exchange_current_density_Dualfoil1998

    return param


def chen_2020():
    """
    Returns
    -------
    pybamm.ParameterValues object with values from Chen 2020 and inputs:
    - Electrode active material volume fractions: am_fraction_n, am_fraction_p
    - Electrode Diffusivities: Ds_n, Ds_p, De
    - Electrode Diffusivity: De
    - Electrode reference exchange-current densities: j0_n, j0_p
    - Cation transference number = t_+
    """

    param = pybamm.ParameterValues("Chen2020")

    param["Positive electrode active material volume fraction"] = pybamm.InputParameter(
        "am_fraction_p"
    )
    param["Negative electrode active material volume fraction"] = pybamm.InputParameter(
        "am_fraction_n"
    )

    param["Cation transference number"] = pybamm.InputParameter("t_+")

    param["Negative electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_n")
    param["Positive electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_p")
    param["Electrolyte diffusivity [m2.s-1]"] = bmp.electrolyte_diffusivity_Nyman2008

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = bmp.graphite_LGM50_electrolyte_exchange_current_density_Chen2020
    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = bmp.nmc_LGM50_electrolyte_exchange_current_density_Chen2020

    return param


def mohtat_2020():
    """
    Returns
    -------
    pybamm.ParameterValues object with values from Mohtat 2020 and inputs:
    - Electrode active material volume fractions: am_fraction_n, am_fraction_p
    - Electrode Diffusivities: Ds_n, Ds_p, De
    - Electrode Diffusivity: De
    - Electrode reference exchange-current densities: j0_n, j0_p
    - Cation transference number = t_+
    """

    param = pybamm.ParameterValues("Chen2020")

    param["Positive electrode active material volume fraction"] = pybamm.InputParameter(
        "am_fraction_p"
    )
    param["Negative electrode active material volume fraction"] = pybamm.InputParameter(
        "am_fraction_n"
    )

    param["Cation transference number"] = pybamm.InputParameter("t_+")

    param[
        "Negative electrode diffusivity [m2.s-1]"
    ] = bmp.graphite_diffusivity_PeymanMPM
    param["Positive electrode diffusivity [m2.s-1]"] = bmp.NMC_diffusivity_PeymanMPM
    param["Electrolyte diffusivity [m2.s-1]"] = bmp.electrolyte_diffusivity_PeymanMPM

    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = bmp.graphite_electrolyte_exchange_current_density_PeymanMPM
    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = bmp.NMC_electrolyte_exchange_current_density_PeymanMPM

    return param
