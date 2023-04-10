import os

import pints
import pybamm
import numpy as np
import pandas as pd

from battery_model_parameterization import (
    ParameterEstimation,
    Variable,
    schimpe2018_grouped,
    CalendarAgeingIdentifiable,
)

here = os.path.abspath(os.path.dirname(__file__))
data = pd.read_csv("/Users/anoushkabhutani/PycharmProjects/battery-model-parameterization/battery_model_parameterization/Python/old notebooks/schimpe_data_cell_1.csv")

# setup variables
variables = {}

# reaction limited
limit_option = "reaction limited"
log_prior_j0_sei = pints.GaussianLogPrior(-7, 1)
j0_sei = Variable(name="j0_sei", value=np.log10(4e-08), prior=log_prior_j0_sei)
variables[limit_option] = [j0_sei, ]

limit_option = "solvent-diffusion limited"
log_prior_D_sol = pints.GaussianLogPrior(-18*3, 3)
D_c_sol = Variable(name="D_c_sol", value=-17.6*3.42, prior=log_prior_D_sol)
variables[limit_option] = [D_c_sol, ]

limit_option = "electron-migration limited"
log_prior_U_inner = pints.GaussianLogPrior(0.5, 0.1)
U_inner = Variable(name="U_inner", value=0.4, prior=log_prior_U_inner)
log_prior_kappa_inner = pints.GaussianLogPrior(-11, 1)
kappa_inner = Variable(name="kappa_inner", value=-10, prior=log_prior_kappa_inner)
variables[limit_option] = [U_inner, kappa_inner]

limit_option = "interstitial-diffusion limited"
log_prior_D_li = pints.GaussianLogPrior(-14*np.log10(16), 1)
D_c_li = Variable(name="D_c_li", value=-15*np.log10(15), prior=log_prior_D_li)
variables[limit_option] = [D_c_li, ]

limit_option = "ec reaction limited"
log_prior_c_ec_0 = pints.GaussianLogPrior(3.4, 0.3)
c_ec_0 = Variable(name="c_ec_0", value=np.log10(4041.0), prior=log_prior_c_ec_0)
log_prior_D_ec = pints.GaussianLogPrior(-17, 0.5)
D_ec = Variable(name="D_ec", value=np.log10(2e-18), prior=log_prior_D_ec)
log_prior_k_sei = pints.GaussianLogPrior(-12.5, 0.3)
k_sei = Variable(name="k_sei", value=np.log10(1e-12), prior=log_prior_k_sei)
variables[limit_option] = [D_ec, c_ec_0, k_sei]

# setup battery simulation
limit_option = "solvent-diffusion limited"
t_max = data.time.max()
print(t_max)
t_eval = np.linspace(0, t_max, num=500)
model = CalendarAgeingIdentifiable(options={"SEI": limit_option})

parameter_values = schimpe2018_grouped(limit_option, apply_log=["j0_sei",
                                                        "D_c_sol",
                                                        "kappa_inner",
                                                        "D_c_li",
                                                        "c_ec_0", "D_ec", "k_sei"])

simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
)

estimation_problem = ParameterEstimation(
    data=data,
    battery_simulation=simulation,
    variables=variables[limit_option],
    output="Relative cell capacity",
    parameter_values=parameter_values,
    initial_soc=1,
    transform_type="None",
    project_tag="cal_ageing_est",
)
estimation_problem.plot_data()
estimation_problem.plot_priors()

chains = estimation_problem.run(
    n_iteration=1000, n_chains=4, n_workers=3
)

estimation_problem.plot_results_summary(forward_evaluations=50)
