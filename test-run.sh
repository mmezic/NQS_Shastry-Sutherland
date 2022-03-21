#PBS -N Test
#PBS -A OPEN-23-38
#PBS -q qnvidia
#PBS -l select=4:ncpus=24:mem=8gb
#PBS -l walltime=0:10:00
#PBS -j oe
script="test.py"

# setting the automatical cleaning of the SCRATCH
# trap 'clean_scratch' TERM EXIT
module load NetKet/3.3.2.post1-OpenMPI-4.1.1-CUDA-11.6.0
# module load Python/3.9.6-GCCcore-11.2.0-NetKet
# module load intel/2020a
# create scratch
SCR="/lscratch/$PBS_JOBID"
mkdir -p $SCR
# copy script to the scratch directory
cp $PBS_O_WORKDIR/$script $SCR
# change directory to the sratch
cd $SCR || exit
# run simulation
mpirun -np 4 python $script
# copy results to my directory
cp *.log *.mpack $PBS_O_WORKDIR
# remove files from the scratch
rm *