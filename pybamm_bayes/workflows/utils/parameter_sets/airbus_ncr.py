import pybamm
import numpy as np

def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e

def graphite_mcmb2528_diffusivity_Dualfoil1998(sto, T):
    """
    Graphite MCMB 2528 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 3.9 * 10 ** (-14)
    E_D_s = 42770
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_mcmb2528_ocp_Dualfoil1998(sto):
    """
    Graphite MCMB 2528 Open-circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
    was measured by Chris Bogatu at Telcordia and PolyStor materials, 2000. However,
    we could not find any other records of this measurment.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    """

    u_eq = (
        0.194
        + 1.5 * np.exp(-120.0 * sto)
        + 0.0351 * np.tanh((sto - 0.286) / 0.083)
        - 0.0045 * np.tanh((sto - 0.849) / 0.119)
        - 0.035 * np.tanh((sto - 0.9233) / 0.05)
        - 0.0147 * np.tanh((sto - 0.5) / 0.034)
        - 0.102 * np.tanh((sto - 0.194) / 0.142)
        - 0.022 * np.tanh((sto - 0.9) / 0.0164)
        - 0.011 * np.tanh((sto - 0.124) / 0.0226)
        + 0.0155 * np.tanh((sto - 0.105) / 0.029)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

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
    m_ref = 2 * 10 ** (-5)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def graphite_entropic_change_Moura2016(sto, c_s_max):
    """
    Graphite entropic change in open-circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from Scott Moura's FastDFN code
    [1].

    References
    ----------
    .. [1] https://github.com/scott-moura/fastDFN

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """
    du_dT = (
        -1.5 * (120.0 / c_s_max) * np.exp(-120 * sto)
        + (0.0351 / (0.083 * c_s_max)) * ((np.cosh((sto - 0.286) / 0.083)) ** (-2))
        - (0.0045 / (0.119 * c_s_max)) * ((np.cosh((sto - 0.849) / 0.119)) ** (-2))
        - (0.035 / (0.05 * c_s_max)) * ((np.cosh((sto - 0.9233) / 0.05)) ** (-2))
        - (0.0147 / (0.034 * c_s_max)) * ((np.cosh((sto - 0.5) / 0.034)) ** (-2))
        - (0.102 / (0.142 * c_s_max)) * ((np.cosh((sto - 0.194) / 0.142)) ** (-2))
        - (0.022 / (0.0164 * c_s_max)) * ((np.cosh((sto - 0.9) / 0.0164)) ** (-2))
        - (0.011 / (0.0226 * c_s_max)) * ((np.cosh((sto - 0.124) / 0.0226)) ** (-2))
        + (0.0155 / (0.029 * c_s_max)) * ((np.cosh((sto - 0.105) / 0.029)) ** (-2))
    )

    return du_dT


def airbus_cell_parameters():
    """
    References
    ----------
    .. [1] >  Zhang, Q., Wang, D., Yang, B., Cui, X. & Li, X.
    Electrochemical model of lithium-ion battery for wide frequency range
    applications. Electrochimica Acta 343, 136094 (2020).
    .. [2] >  Liu, G. & Zhang, L. Research on the Thermal Characteristics 
    of an 18650 Lithium-Ion Battery Based on an Electrochemical–Thermal Flow 
    Coupling Model. World Electric Vehicle Journal 12, 250 (2021).
    .. [3] > PyBaMM parameter set: Marquis2019
    """
    c_n_max = 31507  # [2]
    c_p_max = 48000  # [2]
    parameters = {

        # Negative Electrode
        "Maximum concentration in negative electrode [mol.m-3]": c_n_max,
        "Negative electrode thickness [m]": 40e-6,  # [2] 40 \mu m
        "Negative electrode active material volume fraction": 0.384,  # [2]
        # "Negative electrode porosity": 0.2, # [1] electrolyte phase fraction (?)
        "Negative electrode OCP [V]": graphite_mcmb2528_ocp_Dualfoil1998,
        "Negative electrode exchange-current density [A.m-2]": graphite_electrolyte_exchange_current_density_Dualfoil1998, # [3]
        "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Moura2016, # [3]
        "Negative particle radius [m]": 2.5e-6,  # [2] 2.5 \mu m
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode conductivity [S.m-1]": 100,  # [3]
        "Negative electrode diffusivity [m2.s-1]": graphite_mcmb2528_diffusivity_Dualfoil1998, # [3]

        # Separator
        "Separator thickness [m]": 30e-6,  # [2] 30 \mu m

        # Positive electrode
        "Maximum concentration in positive electrode [mol.m-3]": c_p_max,
        "Positive electrode thickness [m]": 55e-6, # [2] 55 \mu m
        "Positive electrode active material volume fraction": 0.42, # [2]
        "Positive particle radius [m]": 0.25e-6, # [2] 0.25 \mu m
        "Positive electrode diffusivity [m2.s-1]": 1.5e-15, # [2]

        # Electrolyte
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Initial concentration in electrolyte [mol.m-3]": 1200.0,
        "Cation transference number": 0.38,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Nyman2008,
        "Thermodynamic factor": 1.0,

        # Cell
        "Electrode height [m]": 64.5e-3, # 64.5 mm
        "Electrode width [m]": 0.578, # https://www.batterydesign.net/cylindrical-cell-electrode-estimation/
        "Lower voltage cut-off [V]": 2.4,
        "Upper voltage cut-off [V]": 4.5,
        "Typical current [A]": 2.9,
        "Current function [A]": 2.9,
        "Nominal cell capacity [A.h]": 2.9,
    }

    parameter_values = pybamm.ParameterValues("NCA_Kim2011")
    parameter_values.update(parameters, check_already_exists=False)
    return parameter_values
