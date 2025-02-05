import os
import glob
import pints
import pybamm
import pandas as pd
import re

from pybamm_bayes import DegradationParameterEstimation, Variable
from pybamm_bayes.workflows.utils.parameter_sets.airbus_nmc import airbus_cell_parameters

here = os.path.abspath(os.path.dirname(__file__))


# Get data path
base_path = "/nfs/turbo/coe-venkvis/ads/wp1-data/RoomTemperature/pybamm_bayes_input"
rpt_paths = glob.glob(
    f"{base_path}/SN-**/RPT_*.csv"
    )

# Load data 
task_id = os.environ.get('SLURM_ARRAY_TASK_ID') or 0
data_path = rpt_paths[int(task_id)]
data = pd.read_csv(data_path)

# Get full voltage range of cycle for eSOH solver
cycle_voltage_range = {
    "V_min" : 2.5,
    "V_max":  4.2
    }

# discharge and rest only
# start_discharge = data[data.current < -0.6].index.values[0]
# data = data[start_discharge:]

# discharge only
data =  data[data.current < -0.6]

# Get cell and cycle number from data path
cell = re.search(r"SN-\d\d", data_path).group(0)
rpt_num =re.search(r"RPT_\d{1,5}", data_path).group(0)

data["time"] = data["time"].values - data["time"].values[0]

current = pybamm.Interpolant(x = data.time.values, y = -1*data.current.values, children=pybamm.t)

# Setup variables and define priors
prior_eps_n = pints.UniformLogPrior(10*0.4, upper=10*0.9)
prior_eps_p = pints.UniformLogPrior(10*0.4, upper=10*0.9)
prior_Q_Li = pints.UniformLogPrior(2.0, upper=7.0)

# prior_eps_n = pints.TruncatedGaussianLogPrior(0.6, sd=0.1, a=0.4, b=0.9)
# prior_eps_p = pints.TruncatedGaussianLogPrior(0.6, sd=0.1, a=0.4, b=0.9)
# prior_Q_Li = pints.TruncatedGaussianLogPrior(5.1, sd=6, a=2.0, b=7.0)

eps_n = Variable(name="eps_n", value=10*0.6628, prior=prior_eps_n, inverse_transform = lambda x : x/10)
eps_p = Variable(name="eps_p", value=10*0.6809, prior=prior_eps_p, inverse_transform = lambda x : x/10)
Q_Li = Variable(name="Q_Li", value=5.1, prior=prior_Q_Li)

variable_pairs = [[eps_n, eps_p], [Q_Li, eps_p], [eps_n, Q_Li]]
variables = variable_pairs[int(task_id)]
variable_names = [v.name for v in variables]

# Setup battery simulation
model = pybamm.lithium_ion.SPMe()
parameter_values = airbus_cell_parameters()
for v in variables:
    if v.name=="eps_n":
        parameter_values["Negative electrode active material volume fraction"] = pybamm.InputParameter("eps_n")
    if v.name=="eps_p":
        parameter_values["Positive electrode active material volume fraction"] = pybamm.InputParameter("eps_p")
parameter_values["Current function [A]"] = current
parameter_values.update(cycle_voltage_range, check_already_exists=False)

simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("safe"),
    parameter_values=parameter_values,
)

identifiability_problem = DegradationParameterEstimation(
    data=data,
    battery_simulation=simulation,
    parameter_values=parameter_values,
    variables=variables,
    output="Terminal voltage [V]",
    project_tag=f"ADSIdentifiability_DChgOnly_{cell}_{rpt_num}",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=5,
    n_iteration=5000,
    n_chains=5,
    sampling_method="MetropolisRandomWalkMCMC"
)

identifiability_problem.plot_results_summary(resample_posterior = False)
