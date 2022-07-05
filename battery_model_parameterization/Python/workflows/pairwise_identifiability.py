import pints
from battery_model_parameterization.Python.variable import Variable
from battery_model_parameterization.Python.identifiability_problem import (
    IdentifiabilityProblem,
)
from battery_model_parameterization.Python.battery_simulation import (
    dfn_constant_current_discharge,
)
from battery_model_parameterization.Python.sampling import run_mcmc

log_prior_Dsn = pints.GaussianLogPrior(-13, 1)
log_prior_j0_n = pints.GaussianLogPrior(-4.26, 1)

Dsn = Variable(name="Ds_n", true_value=-13.45, prior=log_prior_Dsn)
j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)
variables = [Dsn, j0_n]

model, param = dfn_constant_current_discharge(d_rate=0.1)

ten_hours = 60 * 60 * 10

identifiability_problem = IdentifiabilityProblem(
    model,
    variables,
    parameter_values=param,
    transform_type="log10",
    resolution=10,
    timespan=ten_hours,
    noise=0.005,
)
identifiability_problem.plot_data()
identifiability_problem.plot_priors()

run_mcmc(
    identifiability_problem, burnin=500, n_iteration=3000, n_chains=10, n_workers=3
)
