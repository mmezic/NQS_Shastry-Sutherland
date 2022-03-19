#!/bin/bash
#PBS -S /bin/bash
#PBS -N mezera_nk
#PBS -l walltime=73:59:00
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=32gb:os=debian10:cluster=^krux
#PBS -j oe

#arguments loading
# while getopts j:t:i: option
# do
# case "${option}"
# in
# j) JEXCH=${OPTARG};;
# t) TIME=${OPTARG};;
# i) INCLINATION=${OPTARG};;
# esac
# done
# if [ -z "$JEXCH" ]
# then
#         JEXCH="not_given"
# fi
# if [ -z "$TIME" ]
# then
#         TIME="not_given"
# fi
# if [ -z "$INCLINATION" ]
# then
#         INCLINATION="not_given"
# fi


cat $PBS_NODEFILE
echo "ssh " `hostname`
echo "cd $SCRATCHDIR"
echo "PBS_NODEFILE $PBS_NODEFILE"
SOURCE=/storage/praha1/home/mezic/diplomka
echo SOURCE= $SOURCE 
echo "cd " $SCRATCHDIR >>$SOURCE/stroj.txt
echo "ssh " `hostname` >>$SOURCE/stroj.txt
cd $SCRATCHDIR

module add python/python-3.7.7-intel-19.0.4-mgiwa7z
module add py-pip/py-pip-19.3-intel-19.0.4-hudzomi
export OMP_NUM_THREADS=16
FILE=/storage/praha1/home/mezic/diplomka/netket_scripts/main.py
echo FILE=$FILE
python3 $FILE -f config_RBM64-R
echo "end"
cp -r $SCRATCHDIR/* $SOURCE
