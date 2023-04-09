import os

import pints
import pybamm
import numpy as np

from battery_model_parameterization import (
    MCMCIdentifiabilityAnalysis,
    Variable,
    schimpe2018,
    CalendarAgeing,
)

here = os.path.abspath(os.path.dirname(__file__))

# setup variables
log_prior_afn = pints.GaussianLogPrior(0.5, 0.1)
log_prior_afp = pints.GaussianLogPrior(0.5, 0.1)
afn = Variable(name="am_fraction_n", value=0.486, prior=log_prior_afn)
afp = Variable(name="am_fraction_p", value=0.455, prior=log_prior_afp)
variables = [afn, afp]

# setup battery simulation
limit_option = "reaction limited"
t_max = 60 * 60 * 365 * 24 * 10
t_eval = np.linspace(0, t_max, num=300)
model = CalendarAgeing(options={"SEI": limit_option})
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(schimpe2018(), check_already_exists=False)

simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
)

identifiability_problem = MCMCIdentifiabilityAnalysis(
    battery_simulation=simulation,
    times=t_eval,
    parameter_values=parameter_values,
    variables=variables,
    error_axis="y",
    output="Terminal voltage [V]",
    transform_type="None",
    noise=0.001,
    project_tag="cal_volt",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=1,
    n_iteration=2000,
    n_chains=7,
    n_workers=3,
)

identifiability_problem.plot_results_summary(forward_evaluations=300)
