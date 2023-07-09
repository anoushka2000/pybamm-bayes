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


# Statement of need

`PyBaMM-Bayes` is a Python package for ... .
 It enables ... flexibility or ease-of-use in the user-interface. 
 `PyBaMM-Bayes` was designed to be used by engineers, students, academics and industrial researchers and system designers concerned with ... . 

The API for `PyBaMM-Bayes` was designed to provide a simple and efficient extension to the `PyBaMM` [@pybamm] framework allowing users to . 
`PyBaMM-Bayes` 

# Algorithm

The algorithm used for practical identifiability analysis is show in \autoref{fig:0}. 

At present, 



Any of the gradient-free MCMC samplers implemented in `PINTS` [@pints] may be used to sample the likelihood. Alternatively, Likelihood Free Inference (LFI) can be used. LFI is implemented using the BOLFI (Bayesian Optimisation for Likelihood Free Inference) method interfaced via the `ELFI` package [@elfi].
Several postprocessing and visualization tools are also provided for diagnostics and analysis of the results.

# Example

An example of a small pack is included below. A 4p1s configuration is defined with busbar resistance of 1 $m\Omega$ and interconnection resistance of 10 $m\Omega$. The `Chen2020` [@Chen2020] parameter set is used to define the battery cell chemistry which was gathered using an LG M50 cylindrical cell of 21700 format. By default the single particle model `SPM` is used to define the electrochemical battery model system but a suite of others are available [@Marquis2020] and can be configured using a custom simulation.

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

The output for the examples is shown below as a pack summary in \autoref{fig:2} and an example of a cell variable plot showing each battery current in \autoref{fig:3}.


![Pack summary showing the pack terminal voltage and total current. \label{fig:2}](./paper_figures/Figure_2.png)

![An example of individual cell variable data, any variable defined by the `PyBaMM` model should be accessible. \label{fig:3}](./paper_figures/Figure_3.png)

# Acknowledgements

# References