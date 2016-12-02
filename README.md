# poisson-equation

# compile on lomonosov
 - ssh elena777mc_1854@lomonosov.parallel.ru
 - ssh compiler
 - module add openmpi/1.8.4-gcc 
 - git clone https://github.com/elena777mc/poisson-equation.git
 - cd poisson-equation
 - make
 - cp main ~/_scratch

# run on lomonosov
 - ssh elena777mc_1854@lomonosov.parallel.ru
 - module add slurm/2.5.6 
 - module add openmpi/1.8.4-gcc 
 - sbatch -p test -n 128 ompi ./main 1000 1000
 - squeue | grep elena (check task execution)

# compile and run on bluegene (mpi)
 - scp -r ~/superprac/poisson-equation/ edu-cmc-stud16-618-08@bluegene1.hpc.cs.msu.ru:~/
 - ssh edu-cmc-stud16-618-08@bluegene1.hpc.cs.msu.ru
 - cd poisson-equation
 - make 
 - cp main /gpfs/data/edu-cmc-stud16-618-08
 - mpisubmit.bg -n 128 -w 00:05:00 -m smp ./main 1000 1000
 - llmap (check task execution)



 - ssh elena777mc_1854@lomonosov.parallel.ru
 - ssh compiler
 - module add openmpi/1.8.4-gcc 
 - git clone https://github.com/elena777mc/poisson-equation.git
 - cd poisson-equation
 - make
 - cp main ~/_scratch
