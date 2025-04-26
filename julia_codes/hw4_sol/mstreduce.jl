# Example implementation of MSTReduce (Fig. 3(b)) from
# Chan, E., Heimlich, M., Purkayastha, A. and van de Geijn, R. (2007),
# Collective communication: theory, practice, and experience. Concurrency
# Computat.: Pract. Exper., 19: 1749–1783. doi:10.1002/cpe.1206
#
# In this minimum spanning tree (mst) reduce algorithm:
#   - Divide ranks into two (almost) equal group
#   - all ranks in the partition (called srce) reduce data to one rank (root)
#   - recurse on two groups with root and srce being the "roots" of respective
#     groups
#

function mstreduce!(buf, root, mpicomm;
    left = 0, right = MPI.Comm_size(mpicomm)-1)

    # If there is no one else, let's get outta here!
    left == right && return
    # Short hand for:
    #=
    if left == right
    return
    end
    =#

    # Determine the split
    mid = div(left + right, 2) # integer division

    # Whom do I send to?
    srce = (root <= mid) ? right : left
    # Short hand for:
    #=
    if root <= mid
    srce = right;
    else
    srce = left;
    end
    =#

    # Figure out who we are
    mpirank = MPI.Comm_rank(mpicomm)

    # Recursion:
    # I'm in the left group and the root is my new root
    if mpirank <= mid && root <= mid
        mstreduce!(buf, root, mpicomm; left=left, right=mid)
    # I'm in the left group and the srce is my new root
    elseif mpirank <= mid && root > mid
        mstreduce!(buf, srce, mpicomm; left=left, right=mid)
    # I'm in the right group and the srce is my new root
    elseif mpirank > mid && root <= mid
        mstreduce!(buf, srce, mpicomm; left=mid + 1, right=right)
    # I'm in the right group and the root is my new root
    elseif mpirank > mid && root > mid
        mstreduce!(buf, root, mpicomm; left=mid + 1, right=right)
    end

    # If I'm the root or srce send or recv (respectively)
    req = MPI.REQUEST_NULL
    if mpirank == srce
        req = MPI.Isend(buf, root, 7, mpicomm)
    elseif mpirank == root
        # Allocate somewhere to put the data when we get it
        recv_buf = similar(buf)
        MPI.Recv!(recv_buf, srce, 7, mpicomm)
        # update buffer with the received data
        buf .= buf .+ recv_buf
    end


    # Make sure all my sends are done before I get outta dodge
    if req != MPI.REQUEST_NULL
        MPI.Wait!(req)
    end

end
