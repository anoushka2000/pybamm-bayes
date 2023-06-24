import os

import pints
import pybamm
import numpy as np
import pandas as pd

from battery_model_parameterization import (
    ParameterEstimation,
    Variable,
    schimpe2018_limiting,
    CalendarAgeingLimiting,
)

here = os.path.abspath(os.path.dirname(__file__))
cell_num = 15
data_path = f"/Users/anoushkabhutani/Desktop/schimpe_data/cell_{cell_num}.csv"
data = pd.read_csv(data_path)
soc_init = data.soc.values[0]
t_ambient = data.TdegK.values[0]
# print(len(data))
# setup variables
variables = {}

# reaction limited
limit_option = "reaction limited"  # local
log_prior_j0_sei = pints.GaussianLogPrior(-7, 0.8)
j0_sei = Variable(name="j0_sei", value=np.log10(4e-08), prior=log_prior_j0_sei)
variables[limit_option] = j0_sei

limit_option = "solvent-diffusion limited"  # local
log_prior_D_sol = pints.GaussianLogPrior(-17, 0.8)
D_c_sol = Variable(name="D_c_sol", value=-19 + 3.42, prior=log_prior_D_sol)
variables[limit_option] = D_c_sol

limit_option = "interstitial-diffusion limited"  # arjuna
log_prior_D_li = pints.GaussianLogPrior(-14 + np.log10(16), 0.8)
D_c_li = Variable(name="D_c_li", value=-14.5 + np.log10(15), prior=log_prior_D_li)
variables[limit_option] = D_c_li

# setup battery simulation
model = CalendarAgeingLimiting()

parameter_values = schimpe2018_limiting(apply_log=["j0_sei",
                                                   "D_c_sol",
                                                   "D_c_li", ])
parameter_values.update({"Ambient temperature [K]": t_ambient})

simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
)

solution = simulation.solve(t_eval=data.time.values, inputs={"j0_sei": j0_sei.value,
                                                             "D_c_sol": D_c_sol.value,
                                                             "D_c_li": D_c_li.value})

estimation_problem = ParameterEstimation(
    data=data,
    battery_simulation=simulation,
    variables=list(variables.values()),
    output="Relative cell capacity",
    parameter_values=parameter_values,
    initial_soc=soc_init,
    transform_type="None",
    project_tag=f"cell_{cell_num}_{limit_option}",
)
estimation_problem.plot_data()
estimation_problem.plot_priors()

chains = estimation_problem.run(
    n_iteration=8, n_chains=4, n_workers=3
)

estimation_problem.plot_results_summary(forward_evaluations=3)
