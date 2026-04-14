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

mpirun ${NQSV_MPIOPTS} \
  -genv I_MPI_DEBUG 5 \
  -np 2 \
  -genv I_MPI_PIN_DOMAIN=node \
  python3 $PBS_O_WORKDIR/fb_vs_R_Paralell_MPI_modification.py \
  20 60 9 10 0.0 1e-3 \
  #> logfile_debug_single_9_10.txt


#mpirun ${NQSV_MPIOPTS} \
#  -genv I_MPI_DEBUG 5 \
#  -np 2 \
#  -genv I_MPI_PIN_DOMAIN=node \
#  python3 $PBS_O_WORKDIR/fb_vs_R_Paralell_MPI.py \
#  10 20 10 11 0.0 1e-3 \
#  > logfile_debug_single_10_11_20Rnum.txt
