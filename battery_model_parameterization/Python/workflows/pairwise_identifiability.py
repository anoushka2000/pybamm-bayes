import pints
import numpy as np
from battery_model_parameterization.Python.variable import Variable
from battery_model_parameterization.Python.identifiability_problem import (
    IdentifiabilityProblem,
)
from battery_model_parameterization.Python.battery_simulation import (
    dfn_constant_current_discharge,
)
from battery_model_parameterization.Python.sampling import run_mcmc

means = [-14.26, -8.50]
sd = [1, 1]
log_prior_Dsn = pints.GaussianLogPrior(means[0], sd[0])
log_prior_De = pints.GaussianLogPrior(means[1], sd[1])

log_prior = pints.ComposedLogPrior(log_prior_Dsn, log_prior_De)

Dsn = Variable(name="Ds_n", true_value=np.log10(3.54e-14), prior=log_prior_Dsn)
De = Variable(name="De", true_value=np.log10(9.71e-10), prior=log_prior_De)

model, param = dfn_constant_current_discharge(d_rate=0.1)

ten_hours = 60 * 60 * 10

identifiability_problem = IdentifiabilityProblem(
    model,
    variables=[Dsn, De],
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
