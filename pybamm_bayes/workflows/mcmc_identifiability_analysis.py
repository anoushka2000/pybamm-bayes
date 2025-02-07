import os

import pints
import pybamm

from pybamm_bayes import MCMCIdentifiabilityAnalysis, Variable, marquis_2019

here = os.path.abspath(os.path.dirname(__file__))

# setup variables
log_prior_Dsn = pints.GaussianLogPrior(-12, 1)
log_prior_j0_n = pints.GaussianLogPrior(-4, 1)
Dsn = Variable(name="Ds_n", value=-12.45, prior=log_prior_Dsn)
j0_n = Variable(name="j0_n", value=-4.19, prior=log_prior_j0_n)
variables = [Dsn, j0_n]

# setup battery simulation
model = pybamm.lithium_ion.SPMe()
parameter_values = marquis_2019(variables)
simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
)

# simulation.solve(inputs={
#     "Ds_n": 2.338346895986477e-17,
#     "j0_n": 2.5285395765783075e-06,
# })

# exit()

identifiability_problem = MCMCIdentifiabilityAnalysis(
    battery_simulation=simulation,
    parameter_values=parameter_values,
    variables=variables,
    output="Terminal voltage [V]",
    transform_type="log10",
    noise=0.005,
    project_tag="TEST_LOGS",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=10,
    n_iteration=200,
    n_chains=3,
)

identifiability_problem.plot_results_summary(forward_evaluations=10)
