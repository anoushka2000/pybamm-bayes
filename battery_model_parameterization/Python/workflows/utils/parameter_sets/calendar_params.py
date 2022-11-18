import pybamm


def graphite_ocp_PeymanMPM(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * pybamm.exp(-75 * (sto + 0.001))
        - 0.0120 * pybamm.tanh((sto - 0.127) / 0.016)
        - 0.0118 * pybamm.tanh((sto - 0.155) / 0.016)
        - 0.0035 * pybamm.tanh((sto - 0.220) / 0.020)
        - 0.0095 * pybamm.tanh((sto - 0.190) / 0.013)
        - 0.0145 * pybamm.tanh((sto - 0.490) / 0.020)
        - 0.0800 * pybamm.tanh((sto - 1.030) / 0.055)
    )

    return u_eq


def graphite_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Graphite entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from [1]

    References
    ----------
    .. [1] K.E. Thomas, J. Newman, "Heats of mixing and entropy in porous insertion
           electrode", J. of Power Sources 119 (2003) 844-849

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    du_dT = 10 ** (-3) * (
        0.28
        - 1.56 * sto
        - 8.92 * sto ** (2)
        + 57.21 * sto ** (3)
        - 110.7 * sto ** (4)
        + 90.71 * sto ** (5)
        - 27.14 * sto ** (6)
    )

    return du_dT


def NMC_ocp_PeymanMPM(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open Circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.

    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto**2)
        - 2.0843 * (sto**3)
        + 3.5146 * (sto**4)
        - 2.2166 * (sto**5)
        - 0.5623e-4 * pybamm.exp(109.451 * sto - 100.006)
    )

    return u_eq


def NMC_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Nickel Manganese Cobalt (NMC) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the OCP. The fit is taken from [1].

    References
    ----------
    .. [1] W. Le, I. Belharouak, D. Vissers, K. Amine, "In situ thermal study of
    li1+ x [ni1/ 3co1/ 3mn1/ 3] 1- x o2 using isothermal micro-clorimetric
    techniques",
    J. of the Electrochemical Society 153 (11) (2006) A2147–A2151.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    # Since the equation uses the OCP at each stoichiometry as input,
    # we need OCP function here

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * sto**2
        - 2.0843 * sto**3
        + 3.5146 * sto**4
        - 0.5623 * 10 ** (-4) * pybamm.exp(109.451 * sto - 100.006)
    )

    du_dT = (
        -800 + 779 * u_eq - 284 * u_eq**2 + 46 * u_eq**3 - 2.8 * u_eq**4
    ) * 10 ** (-3)

    return du_dT


# Call dict via a function to avoid errors when editing in place
def calendar_params():

    return {
        # sei
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-10,
        "SEI resistivity [Ohm.m]": 200000.0,
        "SEI solvent diffusivity [m2.s-1]": 2.5e-18,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "SEI electron conductivity [S.m-1]": 8.95e-14,
        "SEI lithium interstitial diffusivity [m2.s-1]": 1e-18,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial SEI thickness [m]": 5e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI open-circuit potential [V]": 0.4,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "SEI growth transfer coefficient": 0.5,
        # cell
        "Negative electrode thickness [m]": 6.2e-05,
        "Positive electrode thickness [m]": 6.7e-05,
        "Electrode height [m]": 1.0,
        "Electrode width [m]": 0.205,
        # negative electrode
        "Maximum concentration in negative electrode [mol.m-3]": 28746.0,
        "Negative electrode OCP [V]": graphite_ocp_PeymanMPM,
        "Negative electrode active material volume fraction": 0.61,
        "Negative particle radius [m]": 2.5e-06,
        "Negative electrode OCP entropic change [V.K-1]"
        "": graphite_entropic_change_PeymanMPM,
        # positive electrode
        "Maximum concentration in positive electrode [mol.m-3]": 35380.0,
        "Positive electrode OCP [V]": NMC_ocp_PeymanMPM,
        "Positive electrode active material volume fraction": 0.445,
        "Positive electrode OCP entropic change [V.K-1]"
        "": NMC_entropic_change_PeymanMPM,
        # electrolyte
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        # experiment
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Initial temperature [K]": 298.15,
        "Initial concentration in negative electrode [mol.m-3]": 48.8682,
        "Initial concentration in positive electrode [mol.m-3]": 31513.0,
        "Lower voltage cut-off [V]": 2.8,
        "Upper voltage cut-off [V]": 4.2,
        # citations
        "citations": ["Mohtat2020"],
    }
