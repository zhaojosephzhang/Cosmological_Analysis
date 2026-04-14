#!/bin/bash
#PBS -N fb30
#PBS --group=hp240141
#PBS -q SQUID
#PBS -l elapstim_req=24:00:00,cpunum_job=60,memsz_job=248GB
#PBS -b 2
#PBS -T intmpi
#PBS -v OMP_NUM_THREADS=60
#PBS -M zzhang@astro-osaka.jp
#PBS -mabe
cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate zhangzhao-env
#module load BaseGCC
#module load BaseCPU
#module load BasePy

# Parameterized inputs for batch submission.
R_MAX="${R_MAX:-20}"
R_NUM="${R_NUM:-180}"
M_LOW="${M_LOW:-14}"
M_UP="${M_UP:-15}"
Z_TARGET="${Z_TARGET:-0.287}"
Z_TOL="${Z_TOL:-1e-2}"
LOG_T="${LOG_T:-4}"
LOG_RHO="${LOG_RHO:-3.5}"
RESUME="${RESUME:-False}"
COMPLETION_MODE="${COMPLETION_MODE:-False}"
COMPLETION_START_SLICE="${COMPLETION_START_SLICE:-63}"
SKIP_HALO_BUILD="${SKIP_HALO_BUILD:-False}"
WRITE_PER_HALO_ROWS="${WRITE_PER_HALO_ROWS:-0}"

mpirun ${NQSV_MPIOPTS} \
  -genv I_MPI_DEBUG 5 \
  -genv WRITE_PER_HALO_ROWS ${WRITE_PER_HALO_ROWS} \
  -np 2 \
  -genv I_MPI_PIN_DOMAIN=node \
  python3 $PBS_O_WORKDIR/fb_vs_R_Paralell_MPI_modification.py \
  ${R_MAX} ${R_NUM} ${M_LOW} ${M_UP} ${Z_TARGET} ${Z_TOL} ${LOG_T} ${LOG_RHO} ${RESUME} ${COMPLETION_MODE} ${COMPLETION_START_SLICE} ${SKIP_HALO_BUILD} \
  #> logfile_debug_single_9_10.txt


#mpirun ${NQSV_MPIOPTS} \
#  -genv I_MPI_DEBUG 5 \
#  -np 2 \
#  -genv I_MPI_PIN_DOMAIN=node \
#  python3 $PBS_O_WORKDIR/fb_vs_R_Paralell_MPI.py \
#  10 20 10 11 0.0 1e-3 \
#  > logfile_debug_single_10_11_20Rnum.txt
