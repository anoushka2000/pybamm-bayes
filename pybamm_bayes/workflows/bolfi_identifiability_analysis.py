import elfi
import pybamm
from battery_model_parameterization import BOLFIIdentifiabilityAnalysis, Variable

# define priors for variables being analysed
prior_Ds_n = elfi.Prior("uniform", 0.1, 10, name="Ds_n")
prior_Ds_p = elfi.Prior("uniform", 0.1, 10, name="Ds_p")

# create a `Variable` object for each variable being analysed
# the `value` here is the 'ground truth' used to create synthetic data
# the `value` needs to be in the sampling space
Ds_n = Variable(name="Ds_n", value=3, prior=prior_Ds_n, bounds=(0.1, 10))
Ds_p = Variable(name="Ds_p", value=1, prior=prior_Ds_p, bounds=(0.1, 10))

variables = [Ds_n, Ds_p]


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

    D_ref = 1e-14 * pybamm.InputParameter(
        "Ds_n"
    )  # default = 3.9 * 10 ** (-14) "Ds_p")  # default = 1 * 10 ** (-13)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


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
    D_ref = 1e-13 * pybamm.InputParameter("Ds_p")  # default = 1 * 10 ** (-13)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


model = pybamm.lithium_ion.SPMe()
param = pybamm.ParameterValues("Marquis2019")
param[
    "Negative electrode diffusivity [m2.s-1]"
] = graphite_mcmb2528_diffusivity_Dualfoil1998
param["Positive electrode diffusivity [m2.s-1]"] = lico2_diffusivity_Dualfoil1998

simulation = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
)

identifiability_problem = BOLFIIdentifiabilityAnalysis(
    battery_simulation=simulation,
    variables=variables,
    output="Terminal voltage [V]",
    parameter_values=param,
    transform_type="log10",
    noise=0.001,
    target_resolution=30,
    project_tag="bolfi_example",
)
identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(sampling_iterations=2000)
identifiability_problem.plot_results_summary()
# identifiability_problem.plot_pairwise()
identifiability_problem.plot_acquistion_surface()
