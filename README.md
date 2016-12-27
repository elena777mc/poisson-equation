# poisson-equation

# compile on lomonosov
 - ssh elena777mc_1854@lomonosov.parallel.ru
 - ssh compiler
 - module add openmpi/1.8.4-gcc 
 - module add cuda/6.5.14
 - git clone https://github.com/elena777mc/poisson-equation.git
 - cd poisson-equation
 - git fetch
 - git checkout -b cuda origin/cuda
 - make
 - cp main ~/_scratch

# run on lomonosov
 - ssh elena777mc_1854@lomonosov.parallel.ru
 - module add slurm/2.5.6 
 - module add openmpi/1.8.4-gcc
 - sbatch -p gputest -n 16 --ntasks-per-node=2 --time=00:15:00 ompi ./main 1000 1000
 - squeue | grep elena (check task execution)
