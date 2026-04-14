# fb MPI Pipeline

This directory stores the MPI submission and analysis scripts used to compute
`f_b(<R)` and `f_gas(<R)` for halo samples in cosmological simulations.

## Files

- `fb_vs_R_Paralell_MPI_modification.py`
  Main Python pipeline. It reads snapshot/group data, builds or reuses halo
  KDTree files, computes `fb` and `f_gas`, and writes summary TXT outputs.

- `run_fb_MPI_modification.original.sh`
  Backup of the original PBS submission script before parameterization.

- `submit_fb_grid_zgt0.sh`
  Batch helper that scans all Fiducial snapshots, selects `z > 0`, and submits
  one PBS job per `(snapshot, mass bin)` combination.

## Expected Companion Script

This pipeline is normally used together with:

- `run_fb_MPI_modification.sh`

If it is not present in this directory, add it before sharing or running the
full workflow from GitHub.

## Environment

Typical runtime requirements:

- PBS job scheduler
- MPI (`mpirun`)
- Conda environment `zhangzhao-env`
- Python packages:
  - `numpy`
  - `h5py`
  - `scipy`
  - `joblib`
  - `mpi4py`
  - `matplotlib`

Example environment activation:

```bash
source ~/.bashrc
conda activate zhangzhao-env
```

## Single Job Submission

The parameterized PBS script is expected to run:

```bash
python3 fb_vs_R_Paralell_MPI_modification.py \
  R_max R_num logM_low logM_up z_target z_tol \
  logT logrho resume completion_mode completion_start_slice skip_halo_build
```

Example:

```bash
python3 fb_vs_R_Paralell_MPI_modification.py \
  20 180 14 15 0.287 1e-2 4 3.5 False False 63 False
```

Argument order:

1. `R_max`
2. `R_num`
3. `logM_low`
4. `logM_up`
5. `z_target`
6. `z_tol`
7. `logT`
8. `logrho`
9. `resume`
10. `completion_mode`
11. `completion_start_slice`
12. `skip_halo_build`

## Batch Submission for z > 0

Run from the code directory:

```bash
cd /sqfs/home/z6b340/code
./submit_fb_grid_zgt0.sh
```

This script will:

- scan Fiducial snapshot headers
- keep only snapshots with `z > 0`
- submit jobs for mass bins:
  - `9-10`
  - `10-11`
  - `11-12`
  - `12-13`
  - `13-14`
  - `14-15`

### Common custom usage

```bash
R_MAX=20 R_NUM=180 Z_TOL=1e-2 WRITE_PER_HALO_ROWS=0 ./submit_fb_grid_zgt0.sh
```

### Restrict to a redshift range

```bash
Z_MIN=0.05 Z_MAX=0.30 ./submit_fb_grid_zgt0.sh
```

### Restrict to a single mass bin

```bash
M_LOW_ONLY=12 M_UP_ONLY=13 ./submit_fb_grid_zgt0.sh
```

### Restrict both redshift and mass bin

```bash
Z_MIN=0.05 Z_MAX=0.30 M_LOW_ONLY=12 M_UP_ONLY=13 ./submit_fb_grid_zgt0.sh
```

## Important Environment Variables

- `R_MAX`
- `R_NUM`
- `Z_TOL`
- `LOG_T`
- `LOG_RHO`
- `RESUME`
- `COMPLETION_MODE`
- `COMPLETION_START_SLICE`
- `SKIP_HALO_BUILD`
- `WRITE_PER_HALO_ROWS`
- `Z_MIN`
- `Z_MAX`
- `M_LOW_ONLY`
- `M_UP_ONLY`
- `BASE_PATH_FIDUCIAL`
- `JOB_PREFIX`

## Notes

- Keep `SKIP_HALO_BUILD=False` when `halos_kdtree_*` files do not yet exist.
- Use `WRITE_PER_HALO_ROWS=0` to suppress very large per-halo TXT tables.
- For large `R_max`, the pipeline re-slices the tail of the halo list to avoid
  oversized slow slices.
