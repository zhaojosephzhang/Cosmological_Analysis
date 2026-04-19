# Gird_data_512_connection

This folder contains the **parallel rotating-LoS** implementation of the CROCODILE light-cone construction. Instead of launching many rays from one observer, it builds a regular grid of parallel lines of sight and connects successive boxes while changing the LoS orientation from one connected segment to the next.

## Files

- `Gird_data_512_connection.ipynb`: notebook version with the full diagnostics and panel-based plotting workflow.
- `Gird_data_512_connection.py`: script version that keeps the data-generation and HDF5-export path.

## Method Summary

This method is designed to reduce repeated structures while still using a deterministic parallel-LoS setup inside each box. For every connected segment:
- a LoS direction is chosen for the new box
- a parallel grid of `X_num x Y_num` sightlines is propagated through that box
- the segment is blended into the previous one across a small overlap region
- the cumulative DM profiles are updated and saved

In this implementation the main scientific products are:
- LOS-resolved electron-density histories
- LOS-resolved cumulative DM histories
- flattened redshift and comoving-distance coordinates for the stitched cone

## Input Data

The input data for this workflow are the `grid_data` products generated from **GADGET4-Osaka** simulation outputs. In these files, physical quantities such as density, temperature, mass, and related gas properties are assigned onto regular Cartesian grids at a chosen resolution level. Typical examples are:

- `lv7`: `256^3` grid cells
- `lv8`: `512^3` grid cells
- `lv9`: `1024^3` grid cells

## Dependencies

The notebook and script were developed around the following Python packages:
- `python>=3.10`
- `numpy`
- `scipy`
- `astropy`
- `h5py`
- `matplotlib`
- `joblib`
- `tqdm`

## Usage

```bash
python Gird_data_512_connection.py \
  8 \
  f \
  100 \
  100 \
  400 \
  1e-4
```

The positional arguments correspond to:
- grid refinement level
- AGN feedback tag
- number of transverse samples along the x direction
- number of transverse samples along the y direction
- number of LoS bins per connected box
- density threshold used in the selection logic

## Outputs

The script writes the stitched light-cone data products to HDF5 and also records the generated LoS vectors. The notebook version additionally produces the diagnostic plots and panel summaries used during exploration.

Figure reference for this method:

<p align="center">
<img src="./figures/figure1_parallel_rotating_los.jpg" width="70%">
</p>
