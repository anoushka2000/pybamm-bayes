from battery_model_parameterization.Python.battery_models.variable_functions.current_density_functions import *
from battery_model_parameterization.Python.battery_models.variable_functions.diffusivity_functions import *


def default_dfn():
    """
    DFN model with diffusivities and reference exchange current densities as inputs.

    Returns
    -------
    model: pybamm.models.full_battery_models.lithium_ion.dfn.DFN
    param: pybamm.parameters.parameter_values.ParameterValues
    """
    model = pybamm.lithium_ion.DFN()

    param = model.default_parameter_values
    param[
        "Negative electrode diffusivity [m2.s-1]"
    ] = graphite_mcmb2528_diffusivity_Dualfoil1998
    param["Positive electrode diffusivity [m2.s-1]"] = lico2_diffusivity_Dualfoil1998
    param["Electrolyte diffusivity [m2.s-1]"] = electrolyte_diffusivity_Capiglia1999
    param[
        "Negative electrode exchange-current density [A.m-2]"
    ] = graphite_electrolyte_exchange_current_density_Dualfoil1998
    param[
        "Positive electrode exchange-current density [A.m-2]"
    ] = lico2_electrolyte_exchange_current_density_Dualfoil1998
    return model, param


def default_spme():
    """
    SPMe model with diffusitivies and cation transference number as inputs.
    Returns
    -------
    model: pybamm.models.full_battery_models.lithium_ion.spme.SPMe
    param: pybamm.parameters.parameter_values.ParameterValues
    """
    model = pybamm.lithium_ion.SPMe()

    param = model.default_parameter_values
    param["Negative electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_n")
    param["Positive electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("Ds_p")
    param["Electrolyte diffusivity [m2.s-1]"] = pybamm.InputParameter("De")
    param["Cation transference number"] = pybamm.InputParameter("t_+")
    return model, param
