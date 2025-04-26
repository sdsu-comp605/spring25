# Naive reduce just has the root receive the message from each rank
function naivereduce!(buf, root, mpicomm)
  # Figure out who we are
  mpirank = MPI.Comm_rank(mpicomm)
  # How many total ranks are there
  mpisize = MPI.Comm_size(mpicomm)

  # If I am the root I will receive the message from everyone
  if mpirank == root
    # Allocate somewhere to put the data when we get it
    recv_buf = similar(buf)
    # Loop through ranks and receive messages
    for n = 1:mpisize
      # MPI uses 0 based indexing for ranks
      neighbor = n-1

      # If not myself receive data
      if neighbor != mpirank
        MPI.Recv!(recv_buf, neighbor, 7, mpicomm)
        # update buffer with the received data
        buf .= buf .+ recv_buf
      end
    end
  else # Since we are not the root, we send data to the root
    MPI.Send(buf, root, 7, mpicomm)
  end
end
