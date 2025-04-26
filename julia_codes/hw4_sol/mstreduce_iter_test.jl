using MPI
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

  # call to mstreduce_iter function
  mstreduce_iter!(buf, root, mpicomm)

  # Root should get the sum (in an MST algorithm, other non-root rank buffers might change, so we don't need to check their values)
  if mpirank == root
    exact = sum(0:mpisize-1)
    @assert buf[1] == exact
  end

  # Let's print the reduced value
  if mpirank == root
    print("MPI size: $mpisize \n")
    print("buf (reduced sum): $(buf[1]) \n")
  end

  # Now let's time the execution
  MPI.Barrier(mpicomm)
  time = @elapsed begin
    mstreduce_iter!(buf, root, mpicomm)
    MPI.Barrier(mpicomm)
  end

  # Finally, let's print the runtime information
  if mpirank == root
    print("Execution time:     $time sec \n")
  end

  # shutdown MPI
  MPI.Finalize()
end
