import elfi
import pybamm

from battery_model_parameterization import BOLFIIdentifiabilityAnalysis, Variable, marquis_2019

# define priors for variables being analysed
prior_Ds_n = elfi.Prior("norm", 13, 0.5, name="Ds_n")
prior_Ds_p = elfi.Prior("norm", 12.5, 0.5, name="Ds_p")

# create a `Variable` object for each variables being analysed
# the `value` here is the 'ground truth' used to create synthetic data
# the `value` needs to be in the sampling space
Ds_n = Variable(name="Ds_n", value=13.4, prior=prior_Ds_n, bounds=(11, 14))
Ds_p = Variable(name="Ds_p", value=13, prior=prior_Ds_p, bounds=(11, 14))

variables = [Ds_n, Ds_p]


model = pybamm.lithium_ion.SPMe()
# create parameter set
param = marquis_2019(variables)

simulation = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
)

identifiability_problem = BOLFIIdentifiabilityAnalysis(
    battery_simulation=simulation,
    variables=variables,
    parameter_values=param,
    transform_type="negated_log10",
    noise=0.001,
    target_resolution=30,
    project_tag="group_meeting_bolfi_log",
)
identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(sampling_iterations=2000)
identifiability_problem.plot_results_summary()
identifiability_problem.plot_acquistion_surface()
