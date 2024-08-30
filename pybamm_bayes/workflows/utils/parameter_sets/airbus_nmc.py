import pybamm
import numpy as np


def nmc_LGM50_ocp_Chen2020(sto):
    """
    LG M50 NMC open-circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq


def graphite_LGM50_ocp_Chen2020(sto):
    """
    LG M50 Graphite open-circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )

    return u_eq


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = 2.22e-11  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    F = pybamm.constants.F
    return F * m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5



def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = 1.79e-11  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    return (
        pybamm.constants.F
        * m_ref
        * c_e**0.5
        * c_s_surf**0.5
        * (c_s_max - c_s_surf) ** 0.5
    )

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_diffusivity_Valoen2005(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration, from [1] (eqn 14)

    References
    ----------
    .. [1] Valøen, Lars Ole, and Jan N. Reimers. "Transport properties of LiPF6-based
    Li-ion battery electrolytes." Journal of The Electrochemical Society 152.5 (2005):
    A882-A891.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte diffusivity [m2.s-1]
    """
    # mol/m3 to molar
    c_e = c_e / 1000

    T_g = 229 + 5 * c_e
    D_0 = -5.43 - 54 / (T - T_g)
    D_1 = -0.25

    # cm2/s to m2/s
    # note, in the Valoen paper, ln means log10, so its inverse is 10^x
    # smaller power => large diffusivity => higher overpotential
    return (10 ** (D_0 + D_1 * c_e)) * 1e-4


def electrolyte_conductivity_Valoen2005(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration, from [1]
    (eqn 17)

    References
    ----------
    .. [1] Valøen, Lars Ole, and Jan N. Reimers. "Transport properties of LiPF6-based
    Li-ion battery electrolytes." Journal of The Electrochemical Society 152.5 (2005):
    A882-A891.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Dimensional electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional electrolyte conductivity [S.m-1]
    """
    # mol/m3 to molar
    c_e = c_e / 1000
    # mS/cm to S/m
    return (1e-3 / 1e-2) * (
        c_e
        * (
            (-10.5 + 0.0740 * T - 6.96e-5 * T**2)
            + c_e * (0.668 - 0.0178 * T + 2.80e-5 * T**2)
            + c_e**2 * (0.494 - 8.86e-4 * T)
        )
        ** 2
    )


def airbus_cell_parameters():
    """
        References
        ----------
        .. [1] > J. Sturm, A. Rheinfeld, I. Zilberman, F.B. Spingler, S. Kosch, F. Frie, A. Jossen,
                Modeling and simulation of inhomogeneities in a 18650 nickel-rich, silicon-graphite 
                lithium-ion cell during fast charging.
    """
    c_n_max = 31360  # [1]
    c_p_max = 52500  # [1]
    parameters = {
        # Negative Electrode
        "Maximum concentration in negative electrode [mol.m-3]": c_n_max,
        "Initial concentration in negative electrode [mol.m-3]": c_n_max * 0.76,
        "Negative electrode thickness [m]": 86.7e-6,  # [1]
        "Negative electrode active material volume fraction": 0.5628,  # [1]
        "Negative electrode porosity": 0.5,  # [1]
        "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,  # [?]
        "Negative electrode OCP entropic change [V.K-1]": 0,
        "Negative electrode exchange-current density [A.m-2]": graphite_LGM50_electrolyte_exchange_current_density_fit,
        "Negative particle radius [m]": 11e-6,  # [1]
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,  # [1]
        "Negative electrode Bruggeman coefficient (electrode)": 0.0,  # [?]
        "Negative electrode conductivity [S.m-1]": 100,  # [1]
        "Negative particle diffusivity [m2.s-1]": 7.4e-15,  # [1] smaller => higher eta
        "Negative electrode charge transfer coefficient": 0.5,  # [1]
        "Negative electrode Butler-Volmer transfer coefficient": 0.5,  # [1]
        "Negative electrode double-layer capacity [F.m-2]": 0.2,  # [?]
        "Negative electrode density [kg.m-3]": 2242,  # [1]
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 867,  # [1]
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.04,  # [1]
        # Separator
        "Separator thickness [m]": 12e-6,  # [1]
        "Separator porosity": 0.5,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 1009,
        "Separator specific heat capacity [J.kg-1.K-1]": 1978.2,
        "Separator thermal conductivity [W.m-1.K-1]": 0.33,
        #
        # Positive electrode
        "Positive electrode conductivity [S.m-1]": 3.8,  # [1]
        "Positive electrode OCP entropic change [V.K-1]": 0,
        "Initial concentration in positive electrode [mol.m-3]": c_p_max * 0.39,
        "Maximum concentration in positive electrode [mol.m-3]": c_p_max,  # [1]
        "Positive particle diffusivity [m2.s-1]":  8e-15,  # [1]
        "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,  # [?]
        "Positive electrode porosity": 0.45,  # [1]
        "Positive electrode active material volume fraction": 0.609,  # [1]
        "Positive particle radius [m]": 5.0e-06,  # [1]
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,  # [1]
        "Positive electrode Bruggeman coefficient (electrode)": 0.0,  # [1]
        "Positive electrode Butler-Volmer transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,  # ?
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_LGM50_electrolyte_exchange_current_density_Chen2020,  # [?]
        "Positive electrode density [kg.m-3]": 4870,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 840.1,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.58,
        "Positive electrode thickness [m]": 66.2e-6,  # [1]
        # Electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,  # [?]
        "Cation transference number": 0.38,  # [1]
        "Electrolyte diffusivity [m2.s-1]": (10 **-6.634) * 1e-4,  # [1]
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Valoen2005,  # [1]
        "Thermodynamic factor": 1.0,  # [?]
        # Cell
        "Electrode height [m]": 65e-3,  # 64.5 mm
        "Electrode width [m]": 1.5,  # https://www.batterydesign.net/cylindrical-cell-electrode-estimation/
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 4.2,
        "Typical current [A]": 2.9,
        "Current function [A]": 2.9,
        "Nominal cell capacity [A.h]": 3.5,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Ambient temperature [K]": 298,
        "Reference temperature [K]": 298,
        "Initial temperature [K]": 298,
        "Number of electrodes connected in parallel to make a cell": 1,
        "Number of cells connected in series to make a battery": 1,
    }

    parameter_values = pybamm.ParameterValues(parameters)
    return parameter_values
