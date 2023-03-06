import os

import pints
import pybamm

from battery_model_parameterization import (
    MCMCIdentifiabilityAnalysis,
    Variable,
    marquis_2019,
)

here = os.path.abspath(os.path.dirname(__file__))

# setup variables
log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)
Dsn = Variable(name="Ds_n", value=-13.45, prior=log_prior_Dsn)
j0_n = Variable(name="j0_n", value=-4.698, prior=log_prior_j0_n)
variables = [Dsn, j0_n]

# setup battery simulation
model = pybamm.lithium_ion.DFN()
parameter_values = marquis_2019(variables)
simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
)

identifiability_problem = MCMCIdentifiabilityAnalysis(
    battery_simulation=simulation,
    parameter_values=parameter_values,
    variables=variables,
    error_axis="y",
    output="Terminal voltage [V]",
    transform_type="log10",
    noise=0.005,
    project_tag="test",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=1,
    n_iteration=10,
    n_chains=2,
    n_workers=3,
)

identifiability_problem.plot_results_summary(forward_evaluations=10)
