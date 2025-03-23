#!/bin/bash

#note: run hostname as a command within this script -- no use of srun.
#      because the --cpus-per-task=16 flag is omitted, other jobs may
#      be allowed to run at the same time on the node requested (via
#      specification of node feature -- each node has it's hostname
#      (node1, node2, etc.) configure as a node feature -- so the name
#      can be used to request a specific node -- we do that here.
#
#      the "sleep=30" can be used to verify that the 2 processes that
#      are run, do run simultaneously (due to backgrounding). verification
#      can be done using the "squeue" command while the jobs is running,
#      in order to monitor the length of runtime.
#

#SBATCH --job-name=j_hello
#SBATCH --output=%julia_hello_world.out
#SBATCH --ntasks=16

mpirun -mca btl self julia mpi_hello_world.jl
