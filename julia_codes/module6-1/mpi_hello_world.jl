using MPI

MPI.Init()
comm = MPI.COMM_WORLD

rank = MPI.Comm_rank(comm)
size = MPI.Comm_rank(comm)

println("I'm rank $rank of $size")
