import pybamm
import numpy as np
from battery_model_parameterization.Python.workflows.utils.parameter_sets import (
    schimpe2018,
)
import pandas as pd

pybamm.set_logging_level("INFO")


class CalendarAgeing(pybamm.lithium_ion.BaseModel):
    def __init__(self, options, name="Calendar ageing model"):
        options = options or {}
        options["timescale"] = 1
        super().__init__(options, name)
        self._length_scales = {}

        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        L_sei = pybamm.Variable("SEI thickness [m]")
        L_sei_init = pybamm.Parameter("Initial SEI thickness [m]")

        # TODO: Calculation attempt 1
        Q = pybamm.Variable("Cell capacity [A.h]")
        Q_init = pybamm.Parameter("Nominal cell capacity [A.h]")

        # mol/m3 of lithium lost to SEI
        V_bar_SEI = pybamm.Parameter("SEI partial molar volume [m3.mol-1]")
        eps = pybamm.Parameter("Negative electrode active material volume fraction")
        R = pybamm.Parameter("Negative particle radius [m]")
        a = 3 * eps / R  # [m^-1]
        n_SEI = (
            L_sei * a / V_bar_SEI
        )  # 1/[m3.mol-1] --> dimn moles of sei/ V_bar_SEI --> mol.m-3 of Li in SEI

        # TODO: Calculation attempt 2
        Q_sei_irr = (L_sei - L_sei_init) * V_bar_SEI  # [m]*[m3.mol-1]  --> [A.h]

        # Negative particle concentration as a function of SEI thickness
        c_s_init = pybamm.Parameter(
            "Initial concentration in negative electrode [mol.m-3]"
        )

        c_s = c_s_init - n_SEI  # Li lost from anode forms SEI => (c_s_init - n_SEI)

        # Potentials as a function of concentration
        c_s_max = pybamm.Parameter(
            "Maximum concentration in negative electrode [mol.m-3]"
        )

        # TODO: Calculation attempt 3
        Q_Li_init = pybamm.Parameter("Cyclable lithium capacity [A.h]")
        Q_Li = Q_Li_init - (c_s / c_s_max) * Q_Li_init
        #         Q_Li = ((c_s_init - n_SEI) / c_s_max)*Q_Li_init

        T = pybamm.Parameter("Ambient temperature [K]")
        U_n = param.n.prim.U_dimensional(
            c_s / c_s_max, T
        )  # OCP_n(conc of Li in anode, T)
        delta_phi = U_n  # (phi_s - phi_e - Un = 0) => phi_s - phi_e = Un
        # eta approx 0 for intercalation reaction but not for SEI
        # since kinetics of intercalation is much faster than SEI
        # j_sei + j_interc = j_total = 0
        # j_sei << => j_interc <<0

        # SEI growth rate
        U_sei = pybamm.Parameter("SEI open-circuit potential [V]")
        eta_SEI = delta_phi - U_sei

        # Thermal prefactor for reaction, interstitial and EC models
        F_RT = param.F / (param.R * T)

        # Define alpha_SEI depending on whether it is symmetric or asymmetric. This
        # applies to "reaction limited" and "EC reaction limited"
        if self.options["SEI"].endswith("(asymmetric)"):
            alpha_SEI = pybamm.Parameter("SEI growth transfer coefficient")
        else:
            alpha_SEI = 0.5

        if self.options["SEI"].startswith("reaction limited"):
            j0_sei = pybamm.Parameter("SEI reaction exchange current density [A.m-2]")
            # Scott Marquis thesis (eq. 5.92)
            j_sei = -j0_sei * pybamm.exp(-0.5 * F_RT * eta_SEI)

        elif self.options["SEI"] == "electron-migration limited":
            U_inner = pybamm.Parameter("SEI open-circuit potential [V]")
            kappa_inner = pybamm.Parameter("SEI electron conductivity [S.m-1]")
            # Scott Marquis thesis (eq. 5.94)
            eta_inner = delta_phi - U_inner
            j_sei = kappa_inner * eta_inner / L_sei

        elif self.options["SEI"] == "interstitial-diffusion limited":
            #  diffusion of lithium-ion interstitials
            # (carry electrons to the SEI reaction site) is the limits SEI growth
            D_li = pybamm.Parameter("SEI lithium interstitial diffusivity [m2.s-1]")
            c_li_0 = pybamm.Parameter(
                "Lithium interstitial reference concentration [mol.m-3]"
            )
            # Scott Marquis thesis (eq. 5.96)
            j_sei = -(D_li * c_li_0 * param.F / L_sei) * pybamm.exp(-F_RT * delta_phi)

        elif self.options["SEI"] == "solvent-diffusion limited":
            # solvent molecules in electrolyte are slow to transport
            # through the outer SEI layer to the reaction site
            # hence rate determining
            D_sol = pybamm.Parameter("SEI solvent diffusivity [m2.s-1]")
            c_sol = pybamm.Parameter("Bulk solvent concentration [mol.m-3]")
            # Scott Marquis thesis (eq. 5.91)
            j_sei = -D_sol * c_sol * param.F / L_sei

        elif self.options["SEI"].startswith("ec reaction limited"):
            c_ec_0 = pybamm.Parameter(
                "EC initial concentration in electrolyte [mol.m-3]"
            )
            D_ec = pybamm.Parameter("EC diffusivity [m2.s-1]")
            k_sei = pybamm.Parameter("SEI kinetic rate constant [m.s-1]")
            # we have a linear system for j and c
            #  c = c_0 + j * L * D / F          [1] (eq 11 in the Yang2017 paper)
            #  j = - F * k * c * exp()          [2] (eq 10 in the Yang2017 paper, factor
            #                                        of a is outside the defn of j here)
            # [1] into [2] gives (F cancels in the second terms)
            #  j = - F * k * c_0 * exp() - k * j * L * D * exp()
            # rearrange
            #  j = -F * k * c_0 * exp() / (1 + k * L * D * exp())
            #  c_ec = c_0 - L * D * k * exp() / (1 + k * L * D * exp())
            #       = c_0 / (1 + k * L * D * exp())
            k_exp = k_sei * pybamm.exp(-alpha_SEI * F_RT * eta_SEI)
            L_D = L_sei * D_ec
            c_0 = c_ec_0
            j_sei = -param.F * c_0 * k_exp / (1 + L_D * k_exp)
        elif self.options["SEI"] in ["none", "constant"]:
            j_sei = pybamm.Scalar(0)

        z_sei = pybamm.Parameter("Ratio of lithium moles to SEI moles")

        # LHS: d L_sei/ dt  [dT/ dt = 0]
        # RHS: L_sei, L_sei^-1   [eta_sei, T]
        self.rhs[L_sei] = -V_bar_SEI * j_sei / (param.F * z_sei)
        self.initial_conditions[L_sei] = L_sei_init

        # TODO: Calculation attempt 1 (Scott Marquis thesis (eq. 5.89a)
        SAn = pybamm.Parameter("Electrode area [m2]")
        self.rhs[Q] = (SAn * j_sei)/3600
        self.initial_conditions[Q] = Q_init

        U_p = param.p.prim.U_init_dim

        # U_n = param.n.prim.U_dimensional(c_s / c_s_max, T)
        # c_s = c_s_init - n_SEI
        # n_SEI = L_sei * a / V_bar_SEI
        V = U_p - U_n

        self.variables = {
            "Terminal voltage [V]": V,  # V = U_p - U_n
            #   = param.p.prim.U_init_dim - param.n.prim.U_dimensional(c_s / c_s_max, T)
            #   = param.p.prim.U_init_dim - param.n.prim.U_dimensional((c_s_init - L_sei * a / V_bar_SEI) / c_s_max, T)
            "Negative particle concentration [mol.m-3]": c_s,
            "Ambient temperature [K]": T,
            "SEI thickness [m]": L_sei,
            "SEI current density [A.m-2]": j_sei,
            "SEI volumetric current density [A.m-3]": a * j_sei,
            "SEI reaction overpotential [V]": eta_SEI,
            "Cell capacity [A.h]": Q,  # TODO: Calculation attempt 1
            "Relative cell capacity": Q/Q_init,
            # "Cell capacity [A.h]": pybamm.Parameter("Nominal cell capacity [A.h]")
            # - Q_sei_irr,
            # # TODO: Calculation attempt 2
            # "Current cyclable lithium capacity [A.h]": Q_Li
            # TODO: Calculation attempt 3 (post processing to Q_loss below)
        }

        self.events = [
            pybamm.Event(
                "Minimum negative particle concentration", pybamm.min(c_s) - 1e-6
            )
        ]

    @property
    def default_geometry(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_var_pts(self):
        return {}


def capacity_loss_from_calendar_ageing(cell_data: pd.DataFrame):
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.update(schimpe2018(), check_already_exists=False)
    parameter_values.update({"Ambient temperature [K]": cell_data.TdegK.values[0]})

    limit_option = "reaction limited"
    model = CalendarAgeing(options={"SEI": limit_option})
    simulation = pybamm.Simulation(
        model, parameter_values=parameter_values, solver=pybamm.CasadiSolver("fast")
    )
    t_max = cell_data.t_hrs.max() * 3600
    t_eval = np.linspace(0, t_max, num=500)
    simulation.solve(t_eval, initial_soc=1)
    solution = simulation.solution

    relative_capacity_lst = []
    param = pybamm.LithiumIonParameters()

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

    Vmin = parameter_values["Lower voltage cut-off [V]"]
    Vmax = parameter_values["Upper voltage cut-off [V]"]

    Q_n = parameter_values["Negative electrode capacity [A.h]"]
    Q_p = parameter_values["Positive electrode capacity [A.h]"]

    Q_Li_solution = solution["Current cyclable lithium capacity [A.h]"].entries
    V_solution = solution["Terminal voltage [V]"].entries

    for i in range(len(Q_Li_solution)):
        Q_Li = Q_Li_solution[i]
        #     Vmax = V_solution[i]  # TODO: Calculation attempt 3.b)
        n_Li = (Q_Li * 3600) / param.F.value
        inputs = {"V_min": Vmin, "V_max": Vmax, "C_n": Q_n, "C_p": Q_p, "n_Li": n_Li}

        esoh_sol = esoh_solver.solve(inputs)
        relative_capacity = (
            esoh_sol["C"].entries[0] / parameter_values["Nominal cell capacity [A.h]"]
        )
        relative_capacity_lst.append(relative_capacity)

    return relative_capacity_lst
