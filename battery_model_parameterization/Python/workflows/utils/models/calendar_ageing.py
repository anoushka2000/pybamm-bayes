import pybamm
import numpy as np

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

        # mol/m3 of lithium lost to SEI
        V_bar_SEI = pybamm.Parameter("SEI partial molar volume [m3.mol-1]")
        eps = pybamm.Parameter("Negative electrode active material volume fraction")
        R = pybamm.Parameter("Negative particle radius [m]")
        a = 3 * eps / R
        n_SEI = L_sei * a / V_bar_SEI

        # Negative particle concentration as a function of SEI thickness
        c_s_init = pybamm.Parameter(
            "Initial concentration in negative electrode [mol.m-3]"
        )
        c_s = c_s_init - n_SEI

        # Potentials as a function of concentration
        c_s_max = pybamm.Parameter(
            "Maximum concentration in negative electrode [mol.m-3]"
        )
        T = pybamm.Parameter("Ambient temperature [K]")
        U_n = param.n.prim.U_dimensional(c_s / c_s_max, T)
        delta_phi = U_n

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
            D_li = pybamm.Parameter("SEI lithium interstitial diffusivity [m2.s-1]")
            c_li_0 = pybamm.Parameter(
                "Lithium interstitial reference concentration [mol.m-3]"
            )
            # Scott Marquis thesis (eq. 5.96)
            j_sei = -(D_li * c_li_0 * param.F / L_sei) * pybamm.exp(-F_RT * delta_phi)

        elif self.options["SEI"] == "solvent-diffusion limited":
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

        self.rhs[L_sei] = -V_bar_SEI * j_sei / (param.F * z_sei)
        self.initial_conditions[L_sei] = L_sei_init

        U_p = param.p.prim.U_init_dim
        V = U_p - U_n

        self.variables = {
            "Terminal voltage [V]": V,
            "Negative particle concentration [mol.m-3]": c_s,
            "SEI thickness [m]": L_sei,
            "SEI current density [A.m-2]": j_sei,
            "SEI volumetric current density [A.m-3]": a * j_sei,
            "SEI reaction overpotential [V]": eta_SEI,
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
