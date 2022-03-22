#!/bin/bash
#PBS -S /bin/bash
#PBS -N mezera_nk
#PBS -A OPEN-23-38
#PBS -q qnvidia
#PBS -l walltime=00:05:00
#PBS -l select=4:ncpus=24:mem=8gb
#PBS -j oe


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
export OMP_NUM_THREADS=24
FILE=/home/mezic/netket_scripts/main.py
echo FILE=$FILE
python3 $FILE -f config
echo "end"
cp -r $SCRATCHDIR/*.log $SOURCE
