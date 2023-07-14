import elfi
import pybamm
from pybamm_bayes import (
    BOLFIIdentifiabilityAnalysis,
    marquis_2019,
    Variable
)

# define priors for variables being analysed
prior_Ds_n = elfi.Prior("uniform", 0.1, 10, name="Ds_n")
prior_Ds_p = elfi.Prior("uniform", 0.1, 10, name="Ds_p")

# create a `Variable` object for each variable being analysed
# the `value` here is the 'ground truth' used to create synthetic data
# the `value` needs to be in the sampling space
Ds_n = Variable(name="Ds_n", value=3, prior=prior_Ds_n, bounds=(0.1, 10))
Ds_p = Variable(name="Ds_p", value=1, prior=prior_Ds_p, bounds=(0.1, 10))

variables = [Ds_n, Ds_p]

model = pybamm.lithium_ion.SPMe()
param = marquis_2019(variables=variables)

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

chains = identifiability_problem.run(sampling_iterations=300)
identifiability_problem.plot_results_summary(forward_evaluations=10)
identifiability_problem.plot_pairwise()
identifiability_problem.plot_acquistion_surface()
