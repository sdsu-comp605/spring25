# Example implementation of MSTBcast (Fig. 3(a)) from
# Chan, E., Heimlich, M., Purkayastha, A. and van de Geijn, R. (2007),
# Collective communication: theory, practice, and experience. Concurrency
# Computat.: Pract. Exper., 19: 1749–1783. doi:10.1002/cpe.1206
#
# In this minimum spanning tree (mst) broadcast algorithm:
#   - Divide ranks into two (almost) equal group
#   - root sends data to one rank in other group (called the dest)
#   - recurse on two groups with root and dest being the "roots" of respective
#     groups
#
#  For nine ranks with root = 1 the algorithm would be (letters just represent
#  who sends/recvs data)
#
#  0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
#  ---------------------------------
#    | x |   |   |   |   |   |   |
#    | a |   |   |   |   |   |   | a  1->8
#    | a |   |   | a |   | b |   | b  1->4, 8->6
#    | a | a | c | c | d | d | b | b  1->2, 4->3, 6->5, 8->7
#  a | a | x | x | x | x | x | x | x  1->0
function mstbcast!(buf, root, mpicomm;
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
  dest = (root <= mid) ? right : left
  # Short hand for:
  #=
  if root <= mid
    dest = right;
  else
    dest = left;
  end
  =#

  # Figure out who we are
  mpirank = MPI.Comm_rank(mpicomm)

  # If I'm the root or dest send or recv (respectively)
  req = MPI.REQUEST_NULL
  if mpirank == root
    req = MPI.Isend(buf, dest, 7, mpicomm)
  elseif mpirank == dest
    MPI.Recv!(buf, root, 7, mpicomm)
  end

  # Recursion:
  # I'm in the left group and the root is my new root
  if mpirank <= mid && root <= mid
    mstbcast!(buf, root, mpicomm; left=left, right=mid)
  # I'm in the left group and the dest is my new root
  elseif mpirank <= mid && root > mid
    mstbcast!(buf, dest, mpicomm; left=left, right=mid)
  # I'm in the right group and the dest is my new root
  elseif mpirank > mid && root <= mid
    mstbcast!(buf, dest, mpicomm, left=mid + 1, right=right)
  # I'm in the right group and the root is my new root
  elseif mpirank > mid && root > mid
    mstbcast!(buf, root, mpicomm, left=mid + 1, right=right)
  end

  # Make sure all my sends are done before I get outta dodge
  MPI.Wait!(req)
end
