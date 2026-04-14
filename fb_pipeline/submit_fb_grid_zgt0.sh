#!/bin/bash
#
# Batch submit helper for fb(<R) jobs at z > 0.
#
# What this script does:
# - scans the Fiducial snapshot directory
# - reads each snapshot header to get redshift z
# - keeps only snapshots with z > 0
# - submits one qsub job per (snapshot, mass-bin) combination
# - mass bins are fixed to: 9-10, 10-11, 11-12, 12-13, 13-14, 14-15
#
# Basic usage:
#   cd /sqfs/home/z6b340/code
#   ./submit_fb_grid_zgt0.sh
#
# Typical customized usage:
#   R_MAX=20 R_NUM=180 Z_TOL=1e-2 WRITE_PER_HALO_ROWS=0 ./submit_fb_grid_zgt0.sh
#   Z_MIN=0.05 Z_MAX=0.30 ./submit_fb_grid_zgt0.sh
#   M_LOW_ONLY=12 M_UP_ONLY=13 ./submit_fb_grid_zgt0.sh
#
# Important environment variables:
#   R_MAX                  maximum radius in units of R200
#   R_NUM                  number of radial bins
#   Z_TOL                  redshift matching tolerance passed to the Python job
#   LOG_T                  log10 temperature threshold
#   LOG_RHO                log10 density threshold
#   RESUME                 True/False
#   COMPLETION_MODE        True/False
#   COMPLETION_START_SLICE starting slice index if completion mode is used
#   SKIP_HALO_BUILD        False is recommended when halos_kdtree files do not yet exist
#   WRITE_PER_HALO_ROWS    0 to suppress per-halo rows in the TXT output
#   BASE_PATH_FIDUCIAL     Fiducial snapshot root directory
#   JOB_PREFIX             qsub job-name prefix
#   Z_MIN                  optional lower bound for submitted redshifts
#   Z_MAX                  optional upper bound for submitted redshifts
#   M_LOW_ONLY             optional single mass-bin lower edge
#   M_UP_ONLY              optional single mass-bin upper edge
#
# Notes:
# - This script discovers z from the Fiducial headers, but each submitted job
#   still runs both rank0=noBH and rank1=fiducial inside run_fb_MPI_modification.sh.
# - If a target redshift has only kdtree_group_* files and no halos_kdtree_* files,
#   keep SKIP_HALO_BUILD=False so the per-halo KDTree files can be generated.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_fb_MPI_modification.sh"

# Defaults for the fb run.
R_MAX="${R_MAX:-20}"
R_NUM="${R_NUM:-180}"
Z_TOL="${Z_TOL:-1e-2}"
LOG_T="${LOG_T:-4}"
LOG_RHO="${LOG_RHO:-3.5}"
RESUME="${RESUME:-False}"
COMPLETION_MODE="${COMPLETION_MODE:-False}"
COMPLETION_START_SLICE="${COMPLETION_START_SLICE:-63}"
SKIP_HALO_BUILD="${SKIP_HALO_BUILD:-False}"
WRITE_PER_HALO_ROWS="${WRITE_PER_HALO_ROWS:-0}"

BASE_PATH_FIDUCIAL="${BASE_PATH_FIDUCIAL:-/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output}"
JOB_PREFIX="${JOB_PREFIX:-fbz}"
Z_MIN="${Z_MIN:-}"
Z_MAX="${Z_MAX:-}"
M_LOW_ONLY="${M_LOW_ONLY:-}"
M_UP_ONLY="${M_UP_ONLY:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to discover snapshot redshifts." >&2
  exit 1
fi

mapfile -t Z_LINES < <(
  conda run -n zhangzhao-env python3 - <<'PY'
import h5py
import os
import re

base = os.environ.get("BASE_PATH_FIDUCIAL", "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output")
snapdirs = sorted(
    [d for d in os.listdir(base) if d.startswith("snapdir_")],
    key=lambda x: int(re.search(r"\d+", x).group()),
)

for snapdir in snapdirs:
    snap_num = re.search(r"\d+", snapdir).group()
    header_file = os.path.join(base, snapdir, f"snapshot_{snap_num}.0.hdf5")
    if not os.path.exists(header_file):
        continue
    with h5py.File(header_file, "r") as f:
        a = float(f["Header"].attrs["Time"])
        z = 1.0 / a - 1.0
    if z > 0:
        print(f"{snap_num} {z:.6f}")
PY
)

if [ "${#Z_LINES[@]}" -eq 0 ]; then
  echo "No z>0 snapshots found under ${BASE_PATH_FIDUCIAL}" >&2
  exit 1
fi

mass_lows=(9 10 11 12 13 14)
mass_ups=(10 11 12 13 14 15)

if [ -n "${M_LOW_ONLY}" ] || [ -n "${M_UP_ONLY}" ]; then
  if [ -z "${M_LOW_ONLY}" ] || [ -z "${M_UP_ONLY}" ]; then
    echo "Both M_LOW_ONLY and M_UP_ONLY must be set together." >&2
    exit 1
  fi

  filtered_mass_lows=()
  filtered_mass_ups=()
  for idx in "${!mass_lows[@]}"; do
    if [ "${mass_lows[$idx]}" = "${M_LOW_ONLY}" ] && [ "${mass_ups[$idx]}" = "${M_UP_ONLY}" ]; then
      filtered_mass_lows+=("${mass_lows[$idx]}")
      filtered_mass_ups+=("${mass_ups[$idx]}")
    fi
  done

  if [ "${#filtered_mass_lows[@]}" -eq 0 ]; then
    echo "Requested mass bin [${M_LOW_ONLY}, ${M_UP_ONLY}] is not supported." >&2
    exit 1
  fi

  mass_lows=("${filtered_mass_lows[@]}")
  mass_ups=("${filtered_mass_ups[@]}")
fi

for z_line in "${Z_LINES[@]}"; do
  snap_num="${z_line%% *}"
  z_target="${z_line##* }"

  if [ -n "${Z_MIN}" ]; then
    if ! awk -v z="${z_target}" -v zmin="${Z_MIN}" 'BEGIN { exit !(z >= zmin) }'; then
      continue
    fi
  fi
  if [ -n "${Z_MAX}" ]; then
    if ! awk -v z="${z_target}" -v zmax="${Z_MAX}" 'BEGIN { exit !(z <= zmax) }'; then
      continue
    fi
  fi

  for idx in "${!mass_lows[@]}"; do
    m_low="${mass_lows[$idx]}"
    m_up="${mass_ups[$idx]}"
    job_name="${JOB_PREFIX}_s${snap_num}_m${m_low}${m_up}"

    echo "Submitting ${job_name}: z=${z_target}, M=[${m_low},${m_up}]"
    qsub \
      -N "${job_name}" \
      -v "R_MAX=${R_MAX},R_NUM=${R_NUM},M_LOW=${m_low},M_UP=${m_up},Z_TARGET=${z_target},Z_TOL=${Z_TOL},LOG_T=${LOG_T},LOG_RHO=${LOG_RHO},RESUME=${RESUME},COMPLETION_MODE=${COMPLETION_MODE},COMPLETION_START_SLICE=${COMPLETION_START_SLICE},SKIP_HALO_BUILD=${SKIP_HALO_BUILD},WRITE_PER_HALO_ROWS=${WRITE_PER_HALO_ROWS}" \
      "${RUN_SCRIPT}"
  done
done
