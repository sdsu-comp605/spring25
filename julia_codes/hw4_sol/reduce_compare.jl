using MPI
include("naivereduce.jl")
include("mstreduce.jl")
include("mstreduce_iter.jl")

let
  # Initialize MPI
  MPI.Init()

  # store communicator
  mpicomm = MPI.COMM_WORLD

  # Get some MPI info
  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # Divide all ranks halfway to determine the root
  root = div(mpisize, 2) # Integer division

  # create buffer for the communication
  buf = [mpirank]

  # test the different reduce versions (non-recursive MST, recursive MST, and naivereduce)
  mstreduce_iter!(buf, root, mpicomm)
  mstreduce!(buf, root, mpicomm)
  naivereduce!(buf, root, mpicomm)

  # Let's time them
  mst_iter_t1 = time_ns() # The time_ns() function in Julia returns the current time in nanoseconds
  mstreduce_iter!(buf, root, mpicomm)
  mst_iter_t2 = time_ns()
  mst_t1 = time_ns() # The time_ns() function in Julia returns the current time in nanoseconds
  mstreduce!(buf, root, mpicomm)
  mst_t2 = time_ns()
  nve_t1 = time_ns()
  naivereduce!(buf, root, mpicomm)
  nve_t2 = time_ns()

  mpirank == 0 && print("Elapsed time for the naive algorithm: \n")
  mpirank == 0 && @show (nve_t2 - nve_t1) * 1e-9
  mpirank == 0 && print("Elapsed time for the recursive mst algorithm: \n")
  mpirank == 0 && @show (mst_t2 - mst_t1) * 1e-9
  mpirank == 0 && print("Elapsed time for the iterative (non-recursive) mst algorithm: \n")
  mpirank == 0 && @show (mst_iter_t2 - mst_iter_t1) * 1e-9

  # shutdown MPI
  MPI.Finalize()
end
