using MPI
include("naivebcast.jl")
include("mstbcast.jl")
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

  # have root broadcast message to everyone
  mstbcast!(buf, root, mpicomm)
  naivebcast!(buf, root, mpicomm)

  mst_t1 = time_ns() # The time_ns() function in Julia returns the current time in nanoseconds
  mstbcast!(buf, root, mpicomm)
  mst_t2 = time_ns()
  nve_t1 = time_ns()
  naivebcast!(buf, root, mpicomm)
  nve_t2 = time_ns()

  mpirank == 0 && print("Elapsed time for the naive algorithm: \n")
  mpirank == 0 && @show (nve_t2 - nve_t1) * 1e-9
  mpirank == 0 && print("Elapsed time for the mst algorithm: \n")
  mpirank == 0 && @show (mst_t2 - mst_t1) * 1e-9

  # shutdown MPI
  MPI.Finalize()
end

