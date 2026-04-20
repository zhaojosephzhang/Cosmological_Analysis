# IGM/CGM Evolution Pipeline

This documentation corresponds to the current implementation:

- `f_igm_z_halo_paralell_MPI_type_formal_FoF.py`
- `run_IGM_z_evolve_MPI_type_formal_FoF.sh`

---

## Overview

This pipeline is designed to:

- compute the redshift evolution of `f_IGM` and `f_CGM`
- measure mass fractions of different components, including `gas`, `star`, and `BH`
- classify gas into different phases:
  - cold-dense
  - cold-diffuse
  - hot-dense
  - hot-diffuse
- compare results between the **NoBH** and **Fiducial** simulations

By default, this script uses `R200` and `M200` values provided in the FoF halo catalog.

The code includes an optional module to recompute `R200/M200`, controlled by the internal flag `if_Cal_R200`.  
However, in the current command-line interface, this option is fixed to `False`.

---

## MPI Layout

- `rank 0` processes the **NoBH** simulation
- `rank 1` processes the **Fiducial** simulation
- at least **2 MPI ranks** are required

---

## Command Line

Command format:

```bash
mpirun -np 2 python f_igm_z_halo_paralell_MPI_type_formal_FoF.py [C_R200] [logM_low] [logM_up] [logT] [logrho]

Parameters

| Parameter  | Type    | Default | Description                                                                               |
| ---------- | ------- | ------: | ----------------------------------------------------------------------------------------- |
| `C_R200`   | `int`   |     `1` | Radius multiplier in units of `R200`. For example, `1` means within `1 × R200`.           |
| `logM_low` | `float` |     `0` | Lower halo mass limit in `log10(M200 / Msun)`.                                            |
| `logM_up`  | `float` |  `15.5` | Upper halo mass limit in `log10(M200 / Msun)`.                                            |
| `logT`     | `float` |   `4.0` | Temperature threshold for gas phase classification.                                       |
| `logrho`   | `float` |   `3.5` | Density threshold (logarithmic), defined relative to the mean density (`rho / rho_mean`). |


The simplest example：

```bash
mpirun -np 2 python f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5
```

## Batch Script

Submit the PBS job using:

```bash
qsub run_IGM_z_evolve_MPI_type_formal_FoF.sh

The script internally runs:

```bash
mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node \
  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5
```

## Outputs

This pipeline produces the redshift evolution of the following quantities:

* `f_IGM`
* `f_CGM`
* `f_star`
* `f_BH`
* `f_cold_dense`
* `f_cold_diff`
* `f_hot_dense`
* `f_hot_diff`

## Output Location

Results are written to:

* `storage_path`
* `kdtree_output_path`

### `kdtree_output_path`

This directory stores the **intermediate KDTree-based spatial indexing data**, designed to accelerate particle queries.

Two types of files are generated:

* `kdtree_group_{i}_{j}_{k}.h5`
  These files partition the full simulation box into coarse spatial regions (groups of cells).
  Each file contains particle data (positions and masses) organized by region and particle type.
  This structure significantly reduces the search volume when querying particles around halos, avoiding full-box scans.

* `halos_kdtree_{file_index}_{C_search}RV.h5`
  These files store **halo-centered particle data** extracted from the grouped KDTree structure.
  Each file contains a subset of halos, where halos are:

  * **sorted in descending order of halo mass (`haloMV`)**
  * then **assigned sequential HaloIDs**
  * and finally **split into chunks (`slice_indices`) and stored across multiple HDF5 files**

  Each halo entry includes:

  * particle positions
  * particle masses
  * separated by particle type (gas, DM, star, BH)

This two-level KDTree design enables efficient halo-based analysis by first narrowing down candidate regions and then constructing halo-local particle datasets.

---

### `storage_path`

This directory stores the **final scientific outputs**, including:

* halo gas mass catalogs:

  * `halo_gas_{C_R200}_r200*.hdf5`

* redshift evolution results:

  * `f_IGM_components_final_*.txt`

* intermediate accumulation files:

  * `f_IGM_components_z_*_temp_MPI_FoF.txt`

* diagnostic plots:

  * `f_IGM_CGM_evolution_*.png`
  * `f_subcomponents_fiducial.png`
  * `f_BH_star_fiducial.png`

These outputs represent the final processed results used for scientific analysis and figures.

### Notes on Intermediate Outputs

The KDTree-related outputs stored in `kdtree_output_path` (including both region-based and halo-based files) are **intermediate data products**.

* These files can be safely removed to save disk space.
* However, if they are deleted, the pipeline will **automatically rebuild them** during the next run, provided the same configuration (e.g., box size, region size, `group_size`, `C_search`) is used.
* Rebuilding these structures can be computationally expensive.

If the files are retained and the configuration remains unchanged, the pipeline will:

* **detect that the KDTree files are up-to-date**
* **skip the reconstruction step**
* and proceed directly to the analysis stage

This design enables a trade-off between **storage usage** and **runtime efficiency**.

## File Naming Convention
Output file names include:
- `dataset label: n (NoBH) or f (Fiducial)`
- `C_R200`
- `halo mass range`
- `logT`
- `logrho`

## Notes

- This FoF-based version directly uses`R200` values provided in the halo catalog
- The option to “recompute R200 exists internally”, but is currently disabled in the command-line interface
- Future versions may expose this option as a dedicated command-line parameter
- Both intermediate (temporary) files and final processed outputs are generated
