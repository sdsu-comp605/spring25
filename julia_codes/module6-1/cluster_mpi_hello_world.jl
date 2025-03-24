using MPI

# Initialize MPI
MPI.Init()

comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
host = gethostname()

println("Hey, I'm rank $rank of $size on $host\n")
