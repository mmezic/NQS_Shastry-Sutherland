#!/bin/bash
#PBS -S /bin/bash
#PBS -N mezera_nk
#PBS -A OPEN-23-38
#PBS -q qprod
#PBS -l walltime=11:35:00
#PBS -l select=1:ncpus=36:mem=16gb
#PBS -j oe
###%// select=4:ncpus=36:mpiprocs=36:ompthreads=1:mem=16gb


cat $PBS_NODEFILE
echo "Hostname ssh " `hostname`
echo "skretc $SCRATCHDIR"
echo "PBS_NODEFILE $PBS_NODEFILE"
SOURCE=/home/mezic/netket_scripts
echo SOURCE= $SOURCE 
echo skretc $SCRATCHDIR >>$SOURCE/stroj.txt
echo "Hostname ssh " `hostname` >>$SOURCE/stroj.txt
cd $SCRATCHDIR

module add NetKet/3.3.2.post1-OpenMPI-4.1.1-CUDA-11.6.0
# export OMP_NUM_THREADS=36
FILE=/home/mezic/netket_scripts/main-models.py
echo FILE=$FILE
python3 $FILE -f config-models16barb
echo "end"
cp -r $SCRATCHDIR/*.log $SOURCE
