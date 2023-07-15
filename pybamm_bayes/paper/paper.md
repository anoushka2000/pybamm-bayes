---
title: 'PyBaMMBayes: A Python package for bayesian identifiability analysis and parameter estimation of battery models with PyBaMM'

tags:
  - Python
  - batteries
  - parameterization
  - MCMC

authors:
  - name: Anoushka Bhutani
    orcid: 0000-0001-9004-1137
    affiliation: "1"
  - name: Valentin Sulzer
    orcid: 0000-0002-8687-327X
    affiliation: "2"

affiliations:
 - name: Carnegie Mellon University, Scott Hall 5109, 5000 Forbes Ave, Pittsburgh, PA 15213, United States.
   index: 1
 - name: The Faraday Institution, Quad One, Becquerel Avenue, Harwell Campus, Didcot, OX11 0RA, United Kingdom.
   index: 2

date: 30 July 2023

bibliography: paper.bib

---

# Summary
Batteries are used ubiquitously is modern technology from mobile phones to electrical vehicles. 
Battery controller design, state monitoring, fault diagnosis, thermal management and accelerated research of advanced battery technologies all rely on computational battery models. 
Electrochemical battery models - such as the Doyle Fuller Newman (DFN) and Single Particle Model (SPM) - are among the most commonly used models. 
These models must be specified in terms of tens of parameters to emulate physical batteries with high fidelity.
Direct empirical measurements of these parameters is difficult, expensive and sometimes impossible. 
Numerous algorithms (genetic algorithms, localized sensitivity etc.) have been applied to fit models to experimental battery cycling data (charging and discharging time series).

In order to determine appropriate, physically relevant parameters from data 

# Statement of need
systems of non-linear equations
`PyBaMM-Bayes` is a Python package for nonlinear quantitative identifiability analysis and parameter estimation for electrochemical models. It provides a simple extension to the `PyBaMM` [@pybamm] battery modelling framework and allow users to conduct practical identifiability analysis and fit `PyBaMM` models to experimental battery cycling data. 

The use of bayesian parameterization provides precise uncertainty estimates for each parameter. Any of the gradient-free MCMC samplers implemented in `PINTS` [@pints] may be used to sample the likelihood. Alternatively, Likelihood Free Inference (LFI) can be used. LFI is implemented using the BOLFI (Bayesian Optimisation for Likelihood Free Inference) method interfaced via the `ELFI` package [@elfi].
Several postprocessing and visualization tools are also provided for diagnostics and analysis of the results. The use MCMC sampling counters the problem of finding sub-optimal local minima, which is often encountered when using other optimization algorithms to parameterize non-linear models.
 `PyBaMM-Bayes` was designed to be used by engineers, students, academics and industrial researchers working on designing batteries and battery systems. 




# Examples of use

An example of identifiability analysis on a DFN model carried is included below. The identifiability of `D_sn` (solid state diffusivity in the negative electrode) and `` (the reference exchange current density) is being tested. Gaussian priors are used for both parameters. The Metropolis Hastings algorithm is used to sample the likelihood with 5 chains and 15000 iterations.

```
import os

import pints
import pybamm

from pybamm_bayes import (
    MCMCIdentifiabilityAnalysis,
    Variable,
    marquis_2019,
)

here = os.path.abspath(os.path.dirname(__file__))

# setup variables
log_prior_Dsn = pints.GaussianLogPrior(-12, 1)
log_prior_j0_n = pints.GaussianLogPrior(-4, 1)
Dsn = Variable(name="Ds_n", value=-12.45, prior=log_prior_Dsn)
j0_n = Variable(name="j0_n", value=-4.19, prior=log_prior_j0_n)
variables = [Dsn, j0_n]

# setup battery simulation
model = pybamm.lithium_ion.DFN()
parameter_values = marquis_2019(variables)
simulation = pybamm.Simulation(
    model,
    solver=pybamm.CasadiSolver("fast"),
    parameter_values=parameter_values,
    experiment=pybamm.Experiment(["Discharge at C/10 for 10 hours"]),
)

identifiability_problem = MCMCIdentifiabilityAnalysis(
    battery_simulation=simulation,
    parameter_values=parameter_values,
    variables=variables,
    output="Terminal voltage [V]",
    transform_type="log10",
    noise=0.005, 
    project_tag="example",
)

identifiability_problem.plot_data()
identifiability_problem.plot_priors()

chains = identifiability_problem.run(
    burnin=200,
    n_iteration=15_000,
    n_chains=5,
)


identifiability_problem.plot_results_summary(forward_evaluations=400)
```

The output for the example is shown is the summary plots below.


![Pack summary showing the pack terminal voltage and total current. \label{fig:2}](./paper_figures/Figure_2.png)

# Acknowledgements

# References