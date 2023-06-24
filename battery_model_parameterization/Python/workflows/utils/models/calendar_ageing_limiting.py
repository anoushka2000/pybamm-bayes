import pybamm

pybamm.set_logging_level("INFO")


class CalendarAgeingLimiting(pybamm.lithium_ion.BaseModel):
    def __init__(self, name="Calendar ageing model"):
        options = {"timescale": 1}
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

        # Negative particle concentration as a function of SEI thickness
        c_s_init = pybamm.Parameter(
            "Initial concentration in negative electrode [mol.m-3]"
        )

        c_s = c_s_init - n_SEI  # Li lost from anode forms SEI => (c_s_init - n_SEI)

        # Potentials as a function of concentration
        c_s_max = pybamm.Parameter(
            "Maximum concentration in negative electrode [mol.m-3]"
        )

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

        # alpha_SEI defined for symmetric, applies to "reaction limited"
        alpha_SEI = 0.5

        # reaction limited
        j0_sei_reaction = pybamm.Parameter("SEI reaction exchange current density [A.m-2]")
        # Scott Marquis thesis (eq. 5.92)
        j0_sei_reaction = -j0_sei_reaction * pybamm.exp(-0.5 * F_RT * eta_SEI)

        # interstitial-diffusion limited
        #  diffusion of lithium-ion interstitials
        # (carry electrons to the SEI reaction site) is the limits SEI growth
        # combined D_li "SEI lithium interstitial diffusivity [m2.s-1]"
        # and c_li_0  "Lithium interstitial reference concentration [mol.m-3]"
        D_c_li = pybamm.Parameter("Lithium interstitial parameter [mol.m-1.s-1]")
        # Scott Marquis thesis (eq. 5.96)
        j0_sei_interstital_diff = -(D_c_li * param.F / L_sei) * pybamm.exp(-F_RT * delta_phi)

        # solvent molecules in electrolyte are slow to transport
        # through the outer SEI layer to the reaction site
        # hence rate determining
        D_c_sol = pybamm.Parameter("Solvent parameter [mol.m-1.s-1]")
        # combines D_sol ("SEI solvent diffusivity [m2.s-1]") and
        # c_sol ("Bulk solvent concentration [mol.m-3]")

        # Scott Marquis thesis (eq. 5.91)
        j0_sei_solv_diffusion = -D_c_sol * param.F / L_sei

        # "none", "constant"
        # j_sei_none = pybamm.Scalar(0)

        z_sei = pybamm.Parameter("Ratio of lithium moles to SEI moles")

        # print(j0_sei_reaction, type(j0_sei_reaction))
        j0_sei = pybamm.minimum(j0_sei_reaction, j0_sei_solv_diffusion)
        j_sei = pybamm.minimum(j0_sei, j0_sei_interstital_diff)

        # LHS: d L_sei/ dt  [dT/ dt = 0]
        # RHS: L_sei, L_sei^-1   [eta_sei, T]
        self.rhs[L_sei] = -V_bar_SEI * j_sei / (param.F * z_sei)
        self.initial_conditions[L_sei] = L_sei_init

        SAn = pybamm.Parameter("Electrode area [m2]")
        self.rhs[Q] = (SAn * j_sei) / 3600
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
            "Cell capacity [A.h]": Q,
            "Relative cell capacity": Q / Q_init,
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
