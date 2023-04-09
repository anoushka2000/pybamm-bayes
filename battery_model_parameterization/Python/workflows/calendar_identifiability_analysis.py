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
variables = {}

# reaction limited
limit_option = "reaction limited"
log_prior_j0_sei = pints.GaussianLogPrior(-7, 1)
j0_sei = Variable(name="j0_sei", value=np.log10(4e-08), prior=log_prior_j0_sei)
variables[limit_option] = [j0_sei, ]

limit_option = "solvent-diffusion limited"
log_prior_D_sol = pints.GaussianLogPrior(-18, 0.25)
D_sol = Variable(name="D_sol", value=-17.6, prior=log_prior_D_sol)
log_prior_c_sol = pints.GaussianLogPrior(3, 0.25)
c_sol = Variable(name="c_sol", value=3.42, prior=log_prior_c_sol)
variables[limit_option] = [D_sol, c_sol]

limit_option = "electron-migration limited"
log_prior_U_inner = pints.GaussianLogPrior(0.5, 0.1)
U_inner = Variable(name="U_inner", value=0.4, prior=log_prior_U_inner)
log_prior_kappa_inner = pints.GaussianLogPrior(-10, 1)
kappa_inner = Variable(name="kappa_inner", value=np.log10(8.95e-14), prior=log_prior_kappa_inner)
variables[limit_option] = [U_inner, kappa_inner]

limit_option = "interstitial-diffusion limited"
log_prior_D_li = pints.GaussianLogPrior(-19, 1)
D_li = Variable(name="D_li", value=np.log10(1e-20), prior=log_prior_D_li)
log_prior_c_li_0 = pints.GaussianLogPrior(16, 1)
c_li_0 = Variable(name="c_li_0", value=15.0, prior=log_prior_c_li_0)

limit_option = "ec reaction limited"
log_prior_c_ec_0 = pints.GaussianLogPrior(3.2, 0.1)
c_ec_0 = Variable(name="c_ec_0", value=np.log10(4041.0), prior=log_prior_c_ec_0)
log_prior_D_ec = pints.GaussianLogPrior(-17, 0.5)
D_ec = Variable(name="D_ec", value=np.log10(2e-18), prior=log_prior_D_ec)
log_prior_k_sei = pints.GaussianLogPrior(13, 0.1)
k_sei = Variable(name="k_sei", value=np.log10(1e-12), prior=log_prior_k_sei)


# setup battery simulation
limit_option = "solvent-diffusion limited"
t_max = 5609*3600
t_eval = np.linspace(0, t_max, num=500)
model = CalendarAgeing(options={"SEI": limit_option})

parameter_values = schimpe2018.schimpe2018(limit_option, apply_log=["j0_sei",
                                                        "D_sol", "c_sol",
                                                        "kappa_inner",
                                                        "D_li",
                                                        "c_ec_0", "D_ec", "k_sei"])

simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
)

identifiability_problem = MCMCIdentifiabilityAnalysis(
    battery_simulation=simulation,
    times=t_eval,
    parameter_values=parameter_values,
    variables=variables[limit_option],
    error_axis="y",
    output="Relative cell capacity",
    transform_type="None",
    noise=2e-4,
    project_tag="cal_Q_loss_solvdiff_limited",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=1,
    n_iteration=500,
    n_chains=5,
    n_workers=3,
)

identifiability_problem.plot_results_summary(forward_evaluations=50)
