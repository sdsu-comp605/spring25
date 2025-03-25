using MPI

# a let-end block (https://docs.julialang.org/en/v1/base/base/#let) just keeps us out of global scope, not really necessary here
# but good practice...
let
  MPI.Init()
  mpicomm = MPI.COMM_WORLD

  mpirank = MPI.Comm_rank(mpicomm)
  mpisize = MPI.Comm_size(mpicomm)

  # send to rank to the right (modulo total number of ranks)
  sendto = mod(mpirank + 1, mpisize)

  # receive from rank to the left (modulo total number of ranks)
  recvfrom = mod(mpirank - 1, mpisize)

  # Pack message we're sending into an array
  sendmsg = [recvfrom mpirank sendto]

  # Create an array to receive a message
  recvmsg = similar(sendmsg)
  recvmsg .= 0

  # non-blocking send and recv
  recvreq = MPI.Irecv!(recvmsg, recvfrom, 777, mpicomm)
  sendreq = MPI.Isend(sendmsg, sendto, 777, mpicomm)

  print("$mpirank sent $sendmsg to $sendto and recv $recvmsg from $recvfrom \n")
end
