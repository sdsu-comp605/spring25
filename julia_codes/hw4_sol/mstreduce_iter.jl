# This version still uses a minimum spanning tree (mst) reduce algorithm, but without using recursion. Instead we use:
#   - Rank Rotation:We first “rotate” the rank numbering so that the process identified by root becomes rank 0 in the new indexing.
#   - Binomial Tree Reduce: We use an iterative loop where the “distance” d doubles in each step.
#

function mstreduce_iter!(buf, root, mpicomm)

    # Get some MPI info
    mpirank = MPI.Comm_rank(mpicomm)
    comm_size = MPI.Comm_size(mpicomm)

    # Compute relative rank (relative to the root) to simplify partner calculations
    rel_rank = mod(mpirank - root + comm_size, comm_size)

    d = 1
    while d < comm_size # Each iteration corresponds to 1 level of the binary tree
        if (rel_rank & d) == 0 # The process is a receiver
        # Determine the partner's relative rank.
        partner = rel_rank | d
        if partner < comm_size # If partner exists, receive their data
            # Convert partner's relative rank back to global rank
            partner_global = mod(partner + root, comm_size)
            # Allocate somewhere to put the data when we get it
            temp = similar(buf)
            MPI.Recv!(temp, partner_global, 7, mpicomm)
            # update buffer with the received data
            buf .= buf .+ temp
        end
        else # The process is a sender
        # It sends its data to the process with rank: rel_rank - d
        partner = rel_rank - d
        # Convert partner's relative rank back to global rank
        partner_global = mod(partner + root, comm_size)
        # Send buf using non-blocking send
        MPI.Isend(buf, partner_global, 7, mpicomm)
        break
        end
        d *= 2 # Move to the next level (and double the group size)
    end

end
