import pybamm

MECHANISM_VARIABLES = {
    "none": [],  # no additional inputs to SEI model
    "constant": [],  # no additional inputs to SEI model
    "reaction limited": [
        ("SEI reaction exchange current density [A.m-2]", 'j0_sei')
    ],
    "solvent-diffusion limited": [
        ("SEI solvent diffusivity [m2.s-1]", 'D_sol'),
        ("Bulk solvent concentration [mol.m-3]", 'c_sol'),
    ],
    "electron-migration limited": [
        ("SEI open-circuit potential [V]", 'U_inner'),
        ("SEI electron conductivity [S.m-1]", 'kappa_inner'),
    ],
    "interstitial-diffusion limited": [
        ("SEI lithium interstitial diffusivity [m2.s-1]", 'D_li'),
        ("Lithium interstitial reference concentration [mol.m-3]", 'c_li_0'),
    ],
    "ec reaction limited": [
        ("EC initial concentration in electrolyte [mol.m-3]", 'c_ec_0'),
        ("EC diffusivity [m2.s-1]", 'D_ec'),
        ("SEI kinetic rate constant [m.s-1]", 'k_sei')
    ]
}


def negative_ocp(sto):
    u_eq = (
            0.6379
            + 0.5416 * pybamm.exp(-305.5309 * sto)
            + 0.044 * pybamm.tanh(-(sto - 0.1958) / 0.1088)
            - 0.1978 * pybamm.tanh((sto - 1.0571) / 0.0854)
            - 0.6875 * pybamm.tanh((sto + 0.0117) / 0.0529)
            - 0.0175 * pybamm.tanh((sto - 0.5692) / 0.0875)
    )

    return u_eq


def positive_ocp(sto):
    u_eq = (
            3.4323
            - 0.8428 * pybamm.exp(-80.2493 * (1 - sto) ** 1.3198)
            - 3.2474e-6 * pybamm.exp(20.2645 * (1 - sto) ** 3.8003)
            + 3.2482e-6 * pybamm.exp(20.2646 * (1 - sto) ** 3.7995)
    )
    return u_eq


# Call dict via a function to avoid errors when editing in place
def _schimpe2018():
    return {
        "Ambient temperature [K]": 298.15,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "Capacity lost to SEI in formation [A.h]": 0.28314252184599464,
        "Cell capacity from negative electrode [A.h]": 2.9773834328737756,
        "Cell capacity from positive electrode [A.h]": 3.0036542364361676,
        "Cyclable lithium capacity [A.h]": 3.1653698047167693,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Electrode area [m2]": 0.157,
        "Electrode height [m]": 0.396232255123179,
        "Electrode width [m]": 0.396232255123179,
        "Initial SEI concentration [mol.m-3]": 1119.623940099833,
        "Initial SEI thickness [m]": 7.360490717322975e-08,
        "Initial temperature [K]": 298.15,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Lower voltage cut-off [V]": 2.7081453187338065,
        "Maximum concentration in negative electrode [mol.m-3]": 31400.0,
        "Maximum concentration in positive electrode [mol.m-3]": 22800.0,
        "Maximum stoichiometry in negative electrode": 0.78,
        "Maximum stoichiometry in positive electrode": 0.916,
        "Minimum stoichiometry in negative electrode": 0.0085,
        "Minimum stoichiometry in positive electrode": 0.045,
        "Moles of SEI formed in formation [mol]": 0.010564435611599993,
        "Negative electrode OCP [V]": negative_ocp,
        "Negative electrode OCP entropic change [V.K-1]": 0,
        "Negative electrode active material volume fraction": 0.486,
        "Negative electrode capacity [A.h]": 3.8592137820787755,
        "Negative electrode cation signed stoichiometry": -1.0,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode loading [A.h.cm-2]": 0.002458097950368647,
        "Negative electrode surface area to volume ratio [m-1]": 1458000.0,
        "Negative electrode thickness [m]": 6.01e-05,
        "Negative particle radius [m]": 1e-06,
        "Nominal cell capacity [A.h]": 2.9905188346549716,
        "Number of cells connected in series to make a battery": 1.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Positive electrode OCP [V]": positive_ocp,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        "Positive electrode active material volume fraction": 0.455,
        "Positive electrode capacity [A.h]": 3.448512326562764,
        "Positive electrode cation signed stoichiometry": -1.0,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode loading [A.h.cm-2]": 0.0021965046666004868,
        "Positive electrode surface area to volume ratio [m-1]": 1365000.0,
        "Positive electrode thickness [m]": 7.9e-05,
        "Positive particle radius [m]": 1e-06,
        "Ratio of lithium moles to SEI moles": 2.0,
        "Reference temperature [K]": 298.15,
        "SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        'SEI electron conductivity [S.m-1]': 8.95e-14,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "SEI growth transfer coefficient": 0.5,
        "SEI open-circuit potential [V]": 0.4,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 5e-8,
        "SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Upper voltage cut-off [V]": 3.421882785122582,
    }


def schimpe2018(mechanism: str):
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.update(_schimpe2018(), check_already_exists=False)
    for parameter in MECHANISM_VARIABLES[mechanism]:
        parameter_values[parameter[0]] = pybamm.InputParameter(parameter[1])
    return parameter_values
