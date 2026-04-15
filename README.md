# Cosmological Simulation Analysis based on CROCODILE

This repository contains a collection of analysis scripts built around simulation data from **[CROCODILE](https://sites.google.com/view/crocodilesimulation/)**, based on **GADGET4-Osaka**.

---

## Overview

The main purpose of this repository is practical:

- to help users who already have a specific analysis goal quickly locate relevant scripts  
- to provide a compact overview of commonly used workflows in cosmological simulation data analysis  

Almost all codes in this repository are written in Python.

---

## A Note on Code Style

These scripts were developed primarily for efficient scientific analysis rather than polished software engineering.

That means:

- coding style is often intuitive rather than highly standardized  
- comments may be mixed in Chinese and English  
- naming conventions are not always uniform  
- some scripts are tightly coupled to specific HPC environments (MPI layout, file paths, storage systems)  

Therefore, the code may not always be fully modular or user-friendly. Please excuse that and jsut treat this repository as a **practical research toolbox**.

---

## Repository Structure

This repository is organized into several analysis pipelines and utility modules. Each subdirectory focuses on a specific type of workflow.

### Core Pipelines

- `fb_pipeline/`  
  Baryon-Fraction-related analysis, including radial baryon fraction profiles and batch processing across redshifts  

- `igm_pipeline/`  
  Analysis of redshift evolution of baryonic components, including  
  `f_IGM`, `f_CGM`, and related quantities  

- `halo_dm_pipeline/`  
  Halo-centered analysis tools, including:  
  - density profile calculations  
  - dispersion measure (DM) vs. impact parameter  
  - 1D sightline projections  
  - 2D observer-style maps  

- `halo_subhalo_catalog/`  
  FoF halo and subhalo catalog extraction and validation utilities  

---

### Light-cone & Line-of-Sight Analysis

- `Light-cone_LoS_analysis/`  

  Light-cone construction and line-of-sight (LoS) analysis based on CROCODILE grid data.  
  This module currently includes two complementary strategies:

  1. **Grid-based connection method**  
     (`Grid_data_512_connection/`)  
     - Parallel rotating LoS approach  
     - Connects multiple simulation boxes along structured sightlines  

  2. **Ray-tracing method**  
     (`Ray_tracing/`)  
     - Box-stacking ray tracing  
     - Supports both rotating and non-rotating configurations  
     - Includes CHIME-like sky window sampling (non-rotating mode)  

---

## Intended Usage

These scripts are designed as **working analysis tools**, not a fully unified software package.

Typical use cases include:

- extracting halo-centered local datasets  
- computing radial density and baryon profiles  
- studying DM, CGM, IGM, stellar, and metallicity properties  
- comparing **Fiducial vs. NoBH** simulations  
- generating 1D/2D projected observables  
- constructing light-cone DM products  
- analyzing sightline-to-sightline variations  
- comparing different light-cone construction strategies  

Detailed workflows are described in the `README.md` files inside each subdirectory.

---

## About Portability

Many scripts depend on:

- MPI-based parallel environments  
- HDF5 data formats  
- HPC job schedulers  
- specific storage layouts (e.g., `/sqfs/...`)  
- CROCODILE / GADGET4-Osaka simulation outputs  

While the **scientific logic is reusable**, direct execution may require environment-specific adjustments.

---

## Citation

If you use this repository for scientific results, please consider citing:

**Zhang et al. (2025)**  
https://iopscience.iop.org/article/10.3847/1538-4357/ae00c2  

---

## Final Remarks

This repository is best viewed as a **practical toolbox for cosmological simulation analysis**.

Recommended entry points:

- `halo_subhalo_catalog/` → catalog generation  
- `halo_dm_pipeline/` → halo-centered DM and gas analysis  
- `igm_pipeline/` → baryon evolution analysis  
- `fb_pipeline/` → feedback-related profiles  
- `Light-cone_LoS_analysis/` → light-cone and LoS-based DM studies  

---
