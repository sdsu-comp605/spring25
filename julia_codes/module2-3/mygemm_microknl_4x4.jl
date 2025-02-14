using SIMD
using KernelAbstractions.Extras: @unroll
using StaticArrays

#
# Using a block size of `nr x mr`
#

# Use block matrix multiply with blocks of `C` of size `nr x mr` with an
# vectorized micro kernel
function mygemm_ji_microknl!(C, A, B, ::Val{mr}, ::Val{nr},
    ::Val{vl}) where {mr, nr, vl}
  # Get the matrix sizes
  m, n = size(C)
  _, k = size(A)

  # Check the matrix sizes
  @assert size(C) == (m, n)
  @assert size(A) == (m, k)
  @assert size(B) == (k, n)

  # We assume that the matrices are even divisible by `nr` and `mr`
  @assert mod(m, mr) == 0
  @assert mod(n, nr) == 0

  # start of the block column number
  for j = 1:nr:n
    jrng = j : (j + nr - 1)
    B_panel = @view B[:, jrng]
    # start of the block row number
    for i = 1:mr:m
      irng = i : (i + mr - 1)
      C_tile = @view C[irng, jrng]
      A_panel = @view A[irng, :]

      # microknl_4xnr!(C_tile, A_panel, B_panel, Val(nr)) # @assert mr == 4
      # microknl_mrxnr!(C_tile, A_panel, B_panel, Val(mr), Val(nr))
      microknl_mrxnr_vl!(C_tile, A_panel, B_panel, Val(mr), Val(nr), Val(vl))
    end
  end
end

# pass nr as a Val
function microknl_4xnr!(C, A, B, ::Val{nr}) where {nr}
  T = eltype(C)

  # We need to stride by the column length of the original C
  m = size(parent(C), 1)

  # Alias for vector unit
  VecT = Vec{4, T}

  # Load the columns of the microtile of C
  c = MArray{Tuple{nr}, VecT}(undef)

  @inbounds @unroll for j = 1:nr
    offset = (j - 1) * m * sizeof(T)
    c[j] = vload(VecT, pointer(C) + offset)
  end

  # How many columns do we have
  pend = size(B, 1)

  # Do the rank-one updates for each p
  # (Unroll means unroll the loop by the factor of 4)
  @inbounds @unroll 4 for p = 1:pend
      a_p = vload(VecT, pointer(A) + (p - 1) * m * sizeof(T))
      @unroll for j = 1:nr
        # c[j] = β_{p1} * a_p + c[j]
        β = VecT(B[p, j])
        c[j] = muladd(β, a_p, c[j])
      end
  end

  # Write back what we just computed using vector stores
  @inbounds @unroll for j = 1:nr
    offset = (j - 1) * m * sizeof(T)
    vstore(c[j], pointer(C) + offset)
  end
end

# passing mr and nr as a Val
function microknl_mrxnr!(C, A, B, ::Val{mr}, ::Val{nr}) where {mr, nr}
  T = eltype(C)

  # We need to stride by the column length of the original C
  m = size(parent(C), 1)

  # Alias for vector unit
  VecT = Vec{4, T}
  mr4 = div(mr, 4)

  # Load the columns of the microtile of C
  c = MArray{Tuple{mr4, nr}, VecT}(undef)

  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr4
      offset = ((i - 1) * 4 + (j - 1) * m) * sizeof(T)
      c[i, j] = vload(VecT, pointer(C) + offset)
    end
  end

  # How many columns do we have
  pend = size(B, 1)

  # storage for micro-vectors of A
  a = MVector{mr4, VecT}(undef)

  # Do the rank-one updates for each p
  # (Unroll means unroll the loop by the factor of 4)
  @inbounds @unroll 4 for p = 1:pend
      @unroll for i = 1:mr4
        offset = ((p - 1) * m + (i - 1) * 4) * sizeof(T)
        a[i] = vload(VecT, pointer(A) + offset)
      end
      @unroll for j = 1:nr
        β = VecT(B[p, j])
        @unroll for i = 1:mr4
          c[i, j] = muladd(β, a[i], c[i, j])
        end
      end
  end

  # Write back what we just computed using vector stores
  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr4
      offset = ((i - 1) * 4 + (j - 1) * m) * sizeof(T)
      vstore(c[i, j], pointer(C) + offset)
    end
  end
end

# passing mr, nr, vl as a Val
function microknl_mrxnr_vl!(C, A, B, ::Val{mr}, ::Val{nr},
    ::Val{vl}) where {mr, nr, vl}
  T = eltype(C)

  # We need to stride by the column length of the original C
  m = size(parent(C), 1)

  # Alias for vector unit
  VecT = Vec{vl, T}
  mr_vl = div(mr, vl)

  # Load the columns of the microtile of C
  c = MArray{Tuple{mr_vl, nr}, VecT}(undef)

  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr_vl
      offset = ((i - 1) * vl + (j - 1) * m) * sizeof(T)
      c[i, j] = vload(VecT, pointer(C) + offset)
    end
  end

  # How many columns do we have
  pend = size(B, 1)

  # storage for micro-vectors of A
  a = MVector{mr_vl, VecT}(undef)

  # Do the rank-one updates for each p
  # (Unroll means unroll the loop by the factor of 4)
  @inbounds @unroll 4 for p = 1:pend
      # load pieces of `a`
      @unroll for i = 1:mr_vl
        offset = ((p - 1) * m + (i - 1) * vl) * sizeof(T)
        a[i] = vload(VecT, pointer(A) + offset)
      end
      # do outer product for each piece
      @unroll for j = 1:nr
        β = B[p, j]
        @unroll for i = 1:mr_vl
          c[i, j] = muladd(β, a[i], c[i, j])
        end
      end
  end

  # Write back what we just computed using vector stores
  @inbounds @unroll for j = 1:nr
    @unroll for i = 1:mr_vl
      offset = ((i - 1) * vl + (j - 1) * m) * sizeof(T)
      vstore(c[i, j], pointer(C) + offset)
    end
  end
end


#
# Assuming a block size of 4x4
#

# Use block matrix multiply with blocks of `C` of size `nr x mr` with an
# vectorized micro kernel
function mygemm_ji_microknl_4x4!(C, A, B)
  # Just for now...
  nr = mr = 4

  # Get the matrix sizes
  m, n = size(C)
  _, k = size(A)

  # Check the matrix sizes
  @assert size(C) == (m, n)
  @assert size(A) == (m, k)
  @assert size(B) == (k, n)

  # We assume that the matrices are even divisible by `nr` and `mr`
  @assert mod(m, mr) == 0
  @assert mod(n, nr) == 0

  # start of the block column number
  for j = 1:nr:n
    jrng = j : (j + nr - 1)
    B_panel = @view B[:, jrng]
    # start of the block row number
    for i = 1:mr:m
      irng = i : (i + mr - 1)
      C_tile = @view C[irng, jrng]
      A_panel = @view A[irng, :]

      # microknl_4x4_v1!(C_tile, A_panel, B_panel)
      # microknl_4x4_v2!(C_tile, A_panel, B_panel)
      microknl_4x4_v3!(C_tile, A_panel, B_panel)
    end
  end
end

# Micro kernel using naive arrays (slow, just doing for correctness!)
function microknl_4x4_v1!(C, A, B)
  T = eltype(C)

  # Length of the vector unit
  VR = 4

  # We need to stride by the column length of the original C
  m = size(parent(C), 1)

  VecT = Vec{VR, T}

  # Load the columns of the microtile of C
  c_1 = C[:, 1]
  c_2 = C[:, 2]
  c_3 = C[:, 3]
  c_4 = C[:, 4]

  pend = size(B, 1)

  @inbounds for p = 1:pend
      a_p = A[:, p]

      # c_1 = β_{p1} * a_p + c_1
      β = B[p, 1]
      # c_1 = muladd(β, a_p, c_1)
      c_1 += β * a_p

      # c_2 = β_{p2} * a_p + c_2
      β = B[p, 2]
      # c_2 = muladd(β, a_p, c_2)
      c_2 += β * a_p

      # c_3 = β_{p3} * a_p + c_3
      β = B[p, 3]
      # c_3 = muladd(β, a_p, c_3)
      c_3 += β * a_p

      # c_4 = β_{p4} * a_p + c_4
      β = B[p, 4]
      # c_4 = muladd(β, a_p, c_4)
      c_4 += β * a_p
  end

  # Write back what we just computed using vector stores
  C[:, 1] = c_1
  C[:, 2] = c_2
  C[:, 3] = c_3
  C[:, 4] = c_4
end

# Micro kernel using vector instructions
function microknl_4x4_v2!(C, A, B)
  T = eltype(C)

  # Length of the vector unit
  VR = 4

  # We need to stride by the column length of the original C
  m = size(parent(C), 1)

  # Alias for vector unit
  VecT = Vec{VR, T}

  # Load the columns of the microtile of C
  c_1 = vload(VecT, pointer(C) + 0m * sizeof(T))
  c_2 = vload(VecT, pointer(C) + 1m * sizeof(T))
  c_3 = vload(VecT, pointer(C) + 2m * sizeof(T))
  c_4 = vload(VecT, pointer(C) + 3m * sizeof(T))

  # How many columns do we have
  pend = size(B, 1)

  # Do the rank-one updates for each p
  # (Unroll means unroll the loop by the factor of 4)
  @inbounds @unroll 4 for p = 1:pend
      a_p = vload(VecT, pointer(A) + (p - 1) * m * sizeof(T))

      # c_1 = β_{p1} * a_p + c_1
      β = VecT(B[p, 1])
      c_1 = muladd(β, a_p, c_1)

      # c_2 = β_{p2} * a_p + c_2
      β = VecT(B[p, 2])
      c_2 = muladd(β, a_p, c_2)

      # c_3 = β_{p3} * a_p + c_3
      β = VecT(B[p, 3])
      c_3 = muladd(β, a_p, c_3)

      # c_4 = β_{p4} * a_p + c_4
      β = VecT(B[p, 4])
      c_4 = muladd(β, a_p, c_4)
  end

  # Write back what we just computed using vector stores
  vstore(c_1, pointer(C) + 0m * sizeof(T))
  vstore(c_2, pointer(C) + 1m * sizeof(T))
  vstore(c_3, pointer(C) + 2m * sizeof(T))
  vstore(c_4, pointer(C) + 3m * sizeof(T))
end

# Micro kernel using vector instructions and Static Arrays
function microknl_4x4_v3!(C, A, B)
  T = eltype(C)

  # We need to stride by the column length of the original C
  m = size(parent(C), 1)

  # Alias for vector unit
  VecT = Vec{4, T}

  # Load the columns of the microtile of C
  c = MArray{Tuple{4}, VecT}(undef)

  @inbounds @unroll for j = 1:4
    offset = (j - 1) * m * sizeof(T)
    c[j] = vload(VecT, pointer(C) + offset)
  end

  # How many columns do we have
  pend = size(B, 1)

  # Do the rank-one updates for each p
  # (Unroll means unroll the loop by the factor of 4)
  @inbounds @unroll 4 for p = 1:pend
      a_p = vload(VecT, pointer(A) + (p - 1) * m * sizeof(T))
      @unroll for j = 1:4
        # c[j] = β_{p1} * a_p + c[j]
        β = VecT(B[p, j])
        c[j] = muladd(β, a_p, c[j])
      end
  end

  # Write back what we just computed using vector stores
  @inbounds @unroll for j = 1:4
    offset = (j - 1) * m * sizeof(T)
    vstore(c[j], pointer(C) + offset)
  end
end

#
# NAIVE KERNELS
#

# Use block matrix multiply with blocks of `C` of size `nr x mr`
function mygemm_ji_pji!(C, A, B, mr, nr)
  # Get the matrix sizes
  m, n = size(C)
  _, k = size(A)

  # Check the matrix sizes
  @assert size(C) == (m, n)
  @assert size(A) == (m, k)
  @assert size(B) == (k, n)

  # We assume that the matrices are even divisible by `nr` and `mr`
  @assert mod(m, mr) == 0
  @assert mod(n, nr) == 0

  # start of the block column number
  for j = 1:nr:n
    jrng = j : (j + nr - 1)
    B_panel = @view B[:, jrng]
    # start of the block row number
    for i = 1:mr:m
      irng = i : (i + mr - 1)
      C_tile = @view C[irng, jrng]
      A_panel = @view A[irng, :]

      mygemm_pji!(C_tile, A_panel, B_panel)
    end
  end
end

# Rank one update (repeatedly update all elements of `C`) with outer product
# using axpy with columns of `A`
function mygemm_pji!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for p = 1:k
    for j = 1:n
      for i = 1:m
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end
