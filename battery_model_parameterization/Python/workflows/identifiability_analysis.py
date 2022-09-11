import pints
import pybamm
from battery_model_parameterization import (IdentifiabilityAnalysis,
                                            Variable,
                                            chen_2020,
                                            run_identifiability_analysis)

# define priors for variables being analysed
log_prior_j0_n = pints.GaussianLogPrior(-5.5, 1)
log_prior_j0_p = pints.GaussianLogPrior(-6.5, 1)

# create a `Variable` object variables being analysed
# the `value` here is the 'ground truth' used to create synthetic data
# the `value` needs to be in the sampling space (in this case a `log10` transformation is applied)
j0_n = Variable(name="j0_n", value=-4.698, prior=log_prior_j0_n)
j0_p = Variable(name="j0_p", value=-6.22, prior=log_prior_j0_p)

variables = [j0_n, j0_p]

model = pybamm.lithium_ion.DFN()

# create parameter set
param = chen_2020(variables)

simulation = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
)

identifiability_problem = IdentifiabilityAnalysis(
    battery_simulation=simulation,
    variables=variables,
    parameter_values=param,
    transform_type="log10",
    noise=0.005,
    project_tag="example",
)
identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = run_identifiability_analysis(
    identifiability_problem, burnin=1, n_iteration=5, n_chains=2, n_workers=3
)
