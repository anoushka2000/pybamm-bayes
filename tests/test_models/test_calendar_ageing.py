import pybamm
import numpy as np
from battery_model_parameterization import CalendarAgeing, calendar_params

models = [
    CalendarAgeing({"SEI": sei}, name=sei)
    for sei in [
        "reaction limited",
        # "reaction limited (asymmetric)",
        # "ec reaction limited",
        # "ec reaction limited (asymmetric)",
        # "interstitial-diffusion limited",
        # "solvent-diffusion limited",
        # "electron-migration limited",
    ]
]

parameter_values = pybamm.ParameterValues(calendar_params())
initial_soc = 0.95
x, y = pybamm.lithium_ion.get_initial_stoichiometries(initial_soc, parameter_values)
param = pybamm.LithiumIonParameters()
c_n_max = parameter_values.evaluate(param.n.prim.c_max)
c_p_max = parameter_values.evaluate(param.p.prim.c_max)
parameter_values.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": x * c_n_max,
        "Initial concentration in positive electrode [mol.m-3]": y * c_p_max,
    }
)

sims = []
for model in models:
    sim = pybamm.Simulation(model, parameter_values=parameter_values)

    seconds_per_year = 365 * 24 * 60 * 60

    t_eval = np.linspace(0, 3 * seconds_per_year, 100)

    solver = pybamm.CasadiSolver(dt_max=10 * seconds_per_year)
    sim.solve(t_eval=t_eval, solver=solver)
    sims.append(sim)

pybamm.dynamic_plot(
    sims,
    [
        "Terminal voltage [V]",
        "Negative particle concentration [mol.m-3]",
        "SEI thickness [m]",
        "SEI reaction overpotential [V]",
    ],
)
