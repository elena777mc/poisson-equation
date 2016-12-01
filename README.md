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
 - sbatch -p test -n 128 ompi ./main
 - squeue | grep elena (check task execution)

