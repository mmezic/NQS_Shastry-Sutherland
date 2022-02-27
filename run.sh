#!/bin/bash
#PBS -S /bin/bash
#PBS -N mezera_nk
#PBS -l walltime=73:55:00
#PBS -l select=1:ncpus=32:mem=6gb:scratch_local=15gb:os=debian10:cluster=^krux
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
echo "Hostname " `hostname`
echo "skretc $SCRATCHDIR"
echo "PBS_NODEFILE $PBS_NODEFILE"
SOURCE=/storage/praha1/home/mezic/diplomka
echo SOURCE= $SOURCE 
echo skretc $SCRATCHDIR >>$SOURCE/stroj.txt
echo "Hostname " `hostname` >>$SOURCE/stroj.txt
cd $SCRATCHDIR

module add python/python-3.7.7-intel-19.0.4-mgiwa7z
module add py-pip/py-pip-19.3-intel-19.0.4-hudzomi
export OMP_NUM_THREADS=16
FILE=/storage/praha1/home/mezic/diplomka/netket_scripts/main-mag.py
echo FILE=$FILE
python3 $FILE -f config_h02-myRBM16
echo "end"
cp -r $SCRATCHDIR/* $SOURCE
