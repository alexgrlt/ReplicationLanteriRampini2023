# ReplicationLanteriRampini2023
The package built in this repository replicates the first best results of "Constrained-Efficient Capital Reallocation" (AER, 2023) by Lanteri and Rampini using Julia. 

The paper is about understanding inefficiencies in the new and old capital markets. One of the main insights from the model (unreplicated here -- see failing_code folder under src to find a tentative to replicate it) is that the equilibrium price of old capital is too high because distributive externalities (due to financial constraints of buyers of old capital) exceed collateral externalities (due to the effects of the resale price of capital on collateral constraints of firms).

## Getting started with the replication

The main function from our package can be called with ReplicationLanteriRampini2023.run_model(), it replicates the first best results of the paper and outputs capital levels (both new, old and their combination). However, because this output is not easy to visualize, one can call a function ReplicationLanteriRampini2023.plot_results(), which builds 3 plots for these levels of capital. The plots for this first best result are not impressive: they only highlight that absent constraints, one should have as much old and new capital (since none is more productive than the other), and that capital should depend on being on the low vs high state but *not* on the current worth of the firm.

In short, we recommend running: 

```
using ReplicationLanteriRampini2023
(worth_grid, kNew_fb,kOld_fb,kTotal_fb) = run_model()
plot_results()
```
or (if you don't want to explicitly call the pkg):

```
(worth_grid, kNew_fb,kOld_fb,kTotal_fb) = ReplicationLanteriRampini2023.run_model()
ReplicationLanteriRampini2023.plot_results()
```

or if only interested in the plots:

```
using ReplicationLanteriRampini2023
plot_results()
```
 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://alexgrlt.github.io/ReplicationLanteriRampini2023.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alexgrlt.github.io/ReplicationLanteriRampini2023.jl/dev/)
[![Build Status](https://github.com/alexgrlt/ReplicationLanteriRampini2023.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/alexgrlt/ReplicationLanteriRampini2023.jl/actions/workflows/CI.yml?query=branch%3Amaster)
