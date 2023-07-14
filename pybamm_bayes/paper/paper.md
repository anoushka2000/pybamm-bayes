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
These models consist of  which must be specified in terms of tens of parameters to emulate physical batteries with high fidelity.
Direct empirical measurements of these parameters are difficult, expensive and sometimes impossible. 
Hence, researchers have applied numerous numerous optimisation algorithms (such as genetic algorithms) to fit battery models to experimental battery cycling data (charging and discharging time series).

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
import liionpack as lp
import pybamm

# Generate the netlist
netlist = lp.setup_circuit(Np=4, Ns=1, Rb=1e-3, Rc=1e-2)

# Define some additional variables to output
output_variables = [
    'X-averaged negative particle surface concentration [mol.m-3]',
    'X-averaged positive particle surface concentration [mol.m-3]',
]

# Cycling experiment, using PyBaMM
experiment = pybamm.Experiment([
    "Charge at 5 A for 30 minutes",
    "Rest for 15 minutes",
    "Discharge at 5 A for 30 minutes",
    "Rest for 30 minutes"],
    period="10 seconds")

# PyBaMM battery parameters
parameter_values = pybamm.ParameterValues("Chen2020")

# Solve the pack problem
output = lp.solve(netlist=netlist,
                  parameter_values=parameter_values,
                  experiment=experiment,
                  output_variables=output_variables,
                  initial_soc=0.5)

# Display the results
lp.plot_output(output)

# Draw the circuit at final state
lp.draw_circuit(netlist, cpt_size=1.0, dpi=150, node_spacing=2.5)
```

The output for the example is shown is the summary plots below.


![Pack summary showing the pack terminal voltage and total current. \label{fig:2}](./paper_figures/Figure_2.png)

![An example of individual cell variable data, any variable defined by the `PyBaMM` model should be accessible. \label{fig:3}](./paper_figures/Figure_3.png)

# Acknowledgements

# References