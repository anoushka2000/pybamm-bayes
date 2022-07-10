import pints

from battery_model_parameterization.Python.battery_simulation.model_setup import (
    dfn_constant_current_discharge,
)
from battery_model_parameterization.Python.identifiability_problem import (
    IdentifiabilityProblem,
)
from battery_model_parameterization.Python.sampling import run_mcmc
from battery_model_parameterization.Python.variable import Variable

# mu = -1*(int(abs(true_value)) - 0.5)
log_prior_Dsn = pints.GaussianLogPrior(-12.5, 1)

# mu = -1*(int(abs(true_value)) + 0.5)
log_prior_j0_n = pints.GaussianLogPrior(-5.5, 1)
# log_prior_j0_p = pints.GaussianLogPrior(-6.5, 1)
# log_prior_Dsp = pints.GaussianLogPrior(-13.5, 1)
# log_prior_De = pints.GaussianLogPrior(-9.5, 1)

Dsn = Variable(name="Ds_n", true_value=-13.408, prior=log_prior_Dsn)
j0_n = Variable(name="j0_n", true_value=-4.698, prior=log_prior_j0_n)
# j0_p = Variable(name="j0_p", true_value=-6.22, prior=log_prior_j0_p)
# Dsp = Variable(name="Ds_p", true_value=-13, prior=log_prior_Dsp)
# De = Variable(name="De", true_value=-9.27, prior=log_prior_De)

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

chains = run_mcmc(
    identifiability_problem, burnin=2, n_iteration=3000, n_chains=10, n_workers=3
)
