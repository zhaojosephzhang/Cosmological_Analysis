# Cosmological Simulation Analysis based on CROCODILE

This repository contains a collection of analysis scripts built around simulation data from **[CROCODILE](https://sites.google.com/view/crocodilesimulation/)**, based on **GADGET4-Osaka**.
<<<<<<< HEAD
<<<<<<< HEAD
- `fb_pipeline/`: baryon-fractoin-related analysis scripts
- `halo_subhalo_catalog/`: FoF halo and subhalo catalog extraction plus validation utilities
- `Light-cone_LoS_analysis/`: light-cone and line-of-sight analysis workflows, including the parallel rotating-LoS and box-stacking ray-tracing methods
=======
The main purpose of this repository is practical:

- to help people who already have a specific analysis goal quickly locate useful scripts
- to help people who are learning cosmological simulation data analysis get a fast overview of several commonly used workflows
=======

The main purpose of this repository is practical:

- to help users who already have a specific analysis goal quickly locate relevant scripts
- to help users who are learning cosmological simulation data analysis get a fast overview of several commonly used workflows
>>>>>>> d0e70ef (modified README.md)

Almost all codes in this repository are written in Python.

## A Note on Code Style

<<<<<<< HEAD
These scripts were written primarily for getting scientific analysis done efficiently, not for polished software engineering.
=======
These scripts were written primarily for efficient scientific analysis rather than polished software engineering.
>>>>>>> d0e70ef (modified README.md)

That means:

- coding style is often intuitive rather than highly standardized
- comments are mixed in Chinese and English
- naming conventions are not always uniform
<<<<<<< HEAD
- some scripts are tightly coupled to a specific HPC environment, MPI layout, file paths, and storage conventions

So for readers, the code may not always be very friendly or elegant. Please excuse that.
=======
- some scripts are tightly coupled to specific HPC environments, MPI layouts, file paths, and storage conventions

So for readers, the code may not always be especially elegant or beginner-friendly. Please excuse that.
>>>>>>> d0e70ef (modified README.md)

## What This Repository Provides

This repository is organized into several analysis pipelines and utility groups. Each subdirectory focuses on one type of simulation-data workflow.

### Available Pipelines

- `fb_pipeline/`
  feedback-related analysis scripts, including radial fb-profile calculations and redshift-batch submission workflows

- `igm_pipeline/`
  scripts for redshift evolution of `f_IGM`, `f_CGM`, and related baryonic component fractions

- `halo_dm_pipeline/`
  halo-centered local data extraction, density profile analysis, DM vs impact-parameter analysis, 1D sightline projections, and 2D observer-style mapping

- `halo_subhalo_catalog/`
  FoF halo and subhalo catalog extraction plus validation utilities

<<<<<<< HEAD
## Intended Usage

These scripts are meant as working analysis tools rather than a fully unified software package.
=======
- `Light-cone_LoS_analysis/`
  light-cone and line-of-sight analysis workflows based on CROCODILE grid data. This directory currently includes two complementary strategies:
  1. `Gird_data_512_connection/`: the parallel rotating-LoS method, which connects multiple boxes with a grid of parallel sightlines
  2. `Ray_tracing/`: the box-stacking ray-tracing method, with both non-rotating and rotating-box implementations, plus CHIME-window sampling support in the non-rotating version

## Intended Usage

These scripts are meant as working analysis tools rather than a unified software package.
>>>>>>> d0e70ef (modified README.md)

In practice, the repository is most useful if you want to:

- extract halo-centered local datasets from large simulation outputs
- compute radial density profiles
- study DM, CGM, IGM, stellar, and metallicity-related quantities
- compare Fiducial and NoBH simulations
- generate 1D or 2D halo-centered projected observables
<<<<<<< HEAD

Most workflows are described in the README files inside the corresponding subdirectories.
=======
- construct light-cone DM products and line-of-sight statistics from stitched simulation boxes
- compare different light-cone building strategies for FRB-DM or foreground analyses

Most workflows are described in the `README.md` files inside the corresponding subdirectories.
>>>>>>> d0e70ef (modified README.md)

## About Portability

Many scripts in this repository depend on:

- MPI
- HDF5
- specific HPC job schedulers
<<<<<<< HEAD
- fixed data layouts under `/sqfs/...`
=======
- fixed data layouts under `/sqfs/...` or other local storage systems
>>>>>>> d0e70ef (modified README.md)
- simulation outputs from CROCODILE / GADGET4-Osaka runs

So while the scientific logic can often be reused elsewhere, direct execution may require environment-specific adjustment.

## Citation

If you directly use these codes to produce scientific analysis results, I would appreciate citation of:

**Zhang et al. 2025**

https://iopscience.iop.org/article/10.3847/1538-4357/ae00c2

## Final Remark

This repository is best viewed as a practical toolbox for cosmological simulation analysis.

If you are browsing it for the first time, the recommended starting points are:

- `halo_subhalo_catalog/` for catalog generation and validation
<<<<<<< HEAD
- `halo_dm_pipeline/` for halo-centered gas / DM / stellar analysis
- `igm_pipeline/` for IGM/CGM redshift-evolution analysis
- `fb_pipeline/` for feedback-related radial profile workflows
>>>>>>> eaa8d58 (update pipelines)


=======
- `halo_dm_pipeline/` for halo-centered gas, DM, and stellar analysis
- `igm_pipeline/` for IGM/CGM redshift-evolution analysis
- `fb_pipeline/` for feedback-related radial profile workflows
- `Light-cone_LoS_analysis/` for stitched light-cone construction and LoS-based DM analysis
>>>>>>> d0e70ef (modified README.md)
