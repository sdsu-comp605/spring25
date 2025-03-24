#!/bin/sh

#note: run julia hello-world on a single node

#SBATCH --job-name=jello
#SBATCH --output=%A-jello.out
#SBATCH --ntasks=16

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self

#require tcp over infiniband
export OMPI_MCA_btl_tcp_if_include ib0

mpirun julia hello.jl

#-------------------------------------------------------------
