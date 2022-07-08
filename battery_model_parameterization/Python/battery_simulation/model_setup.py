import pybamm
from battery_model_parameterization.Python.battery_simulation.current_density_functions import *


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


