# Ray_tracing

This folder contains the **box-stacking** implementation of the light-cone DM calculation. The workflow places a single observer inside the periodic simulation box, samples many sightline directions, assigns each radial shell to the nearest available snapshot, and integrates the cumulative FRB dispersion measure.

## Files

- `Ray_tracing_no_roating.ipynb`: notebook front-end for the non-rotating box-stacking calculation.
- `Ray_tracing_no_roating.py`: production script that generates the data products without the extra plotting cells.
- `Ray_tracing_rotating_box.ipynb`: notebook front-end for the rotating-box variant.
- `Ray_tracing_rotating_box.py`: production script for the rotating-box variant.
- `Ray_tracing_script_runner.ipynb`: lightweight control notebook that directly imports the two `.py` scripts and runs them from a small number of configuration cells.

## Methods

### Non-rotating version

The non-rotating version keeps the original periodic box orientation fixed. It supports two sky samplers:
- full-sky Fibonacci directions
- the **CHIME declination window**, enabled with `use_chime_window=True` or `--use-chime-window`

The CHIME window is implemented by sampling directions with
`dec_min_deg <= declination <= dec_max_deg`.

### Rotating-box version

The rotating-box version uses the same box-stacking idea, but with one extra requirement:
- each radial shell is assigned a shared random rotation, and optionally a shared random translation, so repeated structures do not recur along the cone in the same orientation

The transform is shared by all rays inside a shell, which preserves shell-level angular correlations.

## Dependencies

These scripts were written for a standard scientific Python environment and rely on:
- `python>=3.10`
- `numpy`
- `scipy`
- `h5py`

## Input Data Assumptions

The input data for this workflow are the `grid_data` products generated from **GADGET4-Osaka** simulation outputs. In these files, physical quantities such as density, temperature, mass, and related gas properties are assigned onto regular Cartesian grids at a chosen resolution level. Typical examples are:

- `lv7`: `256^3` grid cells
- `lv8`: `512^3` grid cells
- `lv9`: `1024^3` grid cells

Both scripts expect CROCODILE fullbox files with datasets and attributes such as:
- `/Cell/GridPos`
- `/Cell/GasDensity`
- `/Cell/ElectronAbundance`
- `/Header` attributes including `Time` and `Redshift`
- `/Parameters` attributes including `BoxSize` and `HubbleParam`

## Usage

Non-rotating version:

```bash
python Ray_tracing_no_roating.py \
  --base-dir /path/to/Grid_data/L100N1024_NoBH \
  --agn-tag n \
  --level 7 \
  --snap-min 15 \
  --snap-max 20 \
  --nlos 100000 \
  --z-max 1.0 \
  --overlap-bins 16 \
  --n-z-out 300 \
  --use-chime-window \
  --dec-min -9.5 \
  --dec-max 90.0 \
  --verbose
```

Rotating-box version:

```bash
python Ray_tracing_rotating_box.py \
  --base-dir /path/to/Grid_data/L100N1024_Fiducial \
  --agn-tag f \
  --level 8 \
  --snap-min 15 \
  --snap-max 20 \
  --nlos 30000 \
  --z-max 1.0 \
  --n-profile 300 \
  --step-mult 1.0 \
  --overlap-bins 16 \
  --verbose
```

## Outputs

Both scripts write one HDF5 file containing:
- total DM to `z_max`
- cumulative `DM(<z)` profiles for each LoS
- the sky directions `n_hat`
- snapshot metadata and cosmological parameters

Figure reference for this method:

![Figure 2](../figures/figure2_box_stacking.png)
