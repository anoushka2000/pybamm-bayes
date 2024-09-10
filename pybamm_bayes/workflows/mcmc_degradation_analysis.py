import os

import pints
import pybamm
import pandas as pd

from pybamm_bayes import DegradationParameterEstimation, Variable
from pybamm_bayes.workflows.utils.parameter_sets.airbus_nmc import airbus_cell_parameters

here = os.path.abspath(os.path.dirname(__file__))

# Load data 
data = pd.read_csv("/home/abhutani/pybamm-bayes/scratch/SN01_cycle_1.csv")
data["time"] = data["time"].values - data["time"].values[0]

current = pybamm.Interpolant(x = data.time.values, y = -1*data.current.values, children=pybamm.t)

# Setup variables and define priors
# prior_eps_n = pints.UniformLogPrior(0.2, upper=0.95)
# prior_eps_p = pints.UniformLogPrior(0.2, upper=0.95)
# prior_Q_Li = pints.UniformLogPrior(3., upper=5.)

prior_eps_n = pints.TruncatedGaussianLogPrior(0.5, sd=1, a=0.2, b=0.9)
prior_eps_p = pints.TruncatedGaussianLogPrior(0.6, sd=1, a=0.2, b=0.9)
prior_Q_Li = pints.TruncatedGaussianLogPrior(3.5, sd=1, a=1.0, b=8.0)

eps_n = Variable(name="eps_n", value=0.5628, prior=prior_eps_n)
eps_p = Variable(name="eps_p", value=0.609, prior=prior_eps_p)
Q_Li = Variable(name="Q_Li", value=4.76, prior=prior_Q_Li)

variables = [eps_n, eps_p, Q_Li]

# Setup battery simulation
model = pybamm.lithium_ion.SPMe()
parameter_values = airbus_cell_parameters()
parameter_values["Negative electrode active material volume fraction"] = pybamm.InputParameter("eps_n")
parameter_values["Positive electrode active material volume fraction"] = pybamm.InputParameter("eps_p")
parameter_values["Current function [A]"] = current

simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
)

identifiability_problem = DegradationParameterEstimation(
    data=data,
    battery_simulation=simulation,
    parameter_values=parameter_values,
    variables=variables,
    output="Terminal voltage [V]",
    transform_type="None",
    project_tag="ADS_WP1",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=100,
    n_iteration=15000,
    n_chains=5,
)

identifiability_problem.plot_results_summary(forward_evaluations=300)
