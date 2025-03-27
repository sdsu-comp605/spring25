# Naive broadcast just has the root send the message to each rank
function naivebcast!(buf, root, mpicomm)
  # Figure out who we are
  mpirank = MPI.Comm_rank(mpicomm)

  # If I am the root send the message to everyone
  if mpirank == root
    # How many total ranks are there
    mpisize = MPI.Comm_size(mpicomm)

    # Create an array for the requests
    reqs = Array{MPI.Request}(undef, mpisize)

    # Loop through ranks and send message
    for n = 1:mpisize
      # MPI uses 0 based indexing for ranks
      neighbor = n-1

      # If its me jst set my request to NULL (e.g., no-op)
      if neighbor == mpirank
        reqs[n] = MPI.REQUEST_NULL
      else # otherwise send message to neighbor
        reqs[n] = MPI.Isend(buf, neighbor, 7, mpicomm)
      end
    end
    # Wait on all the requests
    MPI.Waitall!(reqs)
  else # Since we are not the root, we receive
    MPI.Recv!(buf, root, 7, mpicomm)
  end
end
