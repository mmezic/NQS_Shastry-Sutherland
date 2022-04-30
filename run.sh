#!/bin/bash
#PBS -S /bin/bash
#PBS -N mezera_nk
#PBS -l walltime=08:59:00
#PBS -l select=1:ncpus=32:mem=16gb:scratch_local=16gb:os=debian10:cluster=^krux
#PBS -j oe


cat $PBS_NODEFILE
echo "ssh " `hostname`
echo "cd $SCRATCHDIR"
echo "PBS_NODEFILE $PBS_NODEFILE"
SOURCE=/storage/praha1/home/mezic/diplomka
echo SOURCE= $SOURCE 
echo "cd " $SCRATCHDIR >>$SOURCE/stroj.txt
echo "ssh " `hostname` >>$SOURCE/stroj.txt
cd $SCRATCHDIR

module add python/python-3.7.7-intel-19.0.4-mgqiwa7z
module add py-pip/py-pip-19.3-intel-19.0.4-hudzomi
export OMP_NUM_THREADS=32
FILE=/storage/praha1/home/mezic/diplomka/netket_scripts/main.py
echo FILE=$FILE
python3 $FILE -f config_RBM16
echo "end"
cp -r $SCRATCHDIR/* $SOURCE
