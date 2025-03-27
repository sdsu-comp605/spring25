using MPI
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

  # check to make sure we got back the right message
  @assert buf[1] == root

  MPI.Barrier(mpicomm)
  time = @elapsed begin
    mstbcast!(buf, root, mpicomm)
    MPI.Barrier(mpicomm)
  end

  # Let's print the execution time:
  # Short hand for:
  #=
  if mpirank == 0
    print("Execution time: ", time,"\n")
  end
  =#
  mpirank == 0 && print("Execution time: ", time,"\n")

  # shutdown MPI
  MPI.Finalize()
end
