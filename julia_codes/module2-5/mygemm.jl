# The idea to put all the types into a struct is due to Maciej Waruszewski
#
#     https://github.com/mwarusz/MyJuliaGEMM

using KernelAbstractions.Extras: @unroll
using SIMD
using StaticArrays

struct Sizes{mc, nc, kc, mr, nr, vl} end

function Base.getproperty(sizes::Sizes{mc, nc, kc, mr, nr, vl},
    sym::Symbol) where {mc, nc, kc, mr, nr, vl}
  if sym == :mc
    return mc
  elseif sym == :nc
    return nc
  elseif sym == :kc
    return kc
  elseif sym == :mr
    return mr
  elseif sym == :nr
    return nr
  elseif sym == :vl
    return vl
  else
    return getfield(sizes, name)
  end
end

function mygemm!(C, A, B, sizes::Sizes)
  # Get the matrix sizes
  m, n = size(C)
  k = size(B, 1)

  mr = sizes.mr
  nr = sizes.nr
  vl = sizes.vl
  mc = sizes.mc
  nc = sizes.nc
  kc = sizes.kc

  # Check the matrix sizes
  @assert size(A) == (m, k)
  @assert size(B, 2) == n

  # We assume that the matrices are even divisible by `nr` and `mr`
  @assert mod(m, mr) == 0
  @assert mod(n, nr) == 0

  # We assume that the blocking perfectly partitions the matrix
  @assert mod(mc, mr) == 0
  @assert mod(nc, nr) == 0

  # Microvector are perfectly by the chosen vector unit
  @assert mod(mr, vl) == 0

  mygemm_JPI_JI!(C, A, B, sizes)
end

function mygemm_JPI_JI!(C, A, B, sizes::Sizes)
  n = size(C, 2)

  nc = sizes.nc
  @inbounds for j1 = 1:nc:n
    j2 = min(n, j1 + nc - 1)
    C_J = @view C[:, j1:j2]
    B_J = @view B[:, j1:j2]
    mygemm_PI_JI!(C_J, A, B_J, sizes)
  end
end

@inline function mygemm_PI_JI!(C, A, B, sizes::Sizes)
  k = size(A, 2)

  kc = sizes.kc
  @inbounds for p1 = 1:kc:k
    p2 = min(k, p1 + kc - 1)
    A_P = @view A[:, p1:p2]
    B_P = @view B[p1:p2, :]
    mygemm_I_JI!(C, A_P, B_P, sizes)
  end
end

@inline function mygemm_I_JI!(C, A, B, sizes::Sizes)
  m = size(C, 1)

  mc = sizes.mc
  @inbounds for i1 = 1:mc:m
    i2 = min(m, i1 + mc - 1)
    C_I = @view C[i1:i2, :]
    A_I = @view A[i1:i2, :]
    mygemm_JI!(C_I, A_I, B, sizes)
  end
end

@inline function mygemm_JI!(C, A, B, sizes::Sizes)
  n = size(C, 2)

  nr = sizes.nr
  @inbounds for j1 = 1:nr:n
    j2 = min(n, j1 + nr - 1)
    C_J = @view C[:, j1:j2]
    B_J = @view B[:, j1:j2]
    mygemm_I!(C_J, A, B_J, sizes)
  end
end

@inline function mygemm_I!(C, A, B, sizes::Sizes)
  m = size(C, 1)

  mr = sizes.mr
  @inbounds for i1 = 1:mr:m
    i2 = min(m, i1 + mr - 1)
    C_I = @view C[i1:i2, :]
    A_I = @view A[i1:i2, :]
    microknl!(C_I, A_I, B, sizes)
  end
end

@inline function microknl!(C, A, B, sizes::Sizes)
  mr = sizes.mr
  nr = sizes.nr
  vl = sizes.vl

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
  @inbounds for p = 1:pend
    # load pieces of `a`
    @unroll for i = 1:mr_vl
      offset = ((p - 1) * m + (i - 1) * vl) * sizeof(T)
      a[i] = vload(VecT, pointer(A) + offset)
    end
    # do outer product for each piece
    @unroll for j = 1:nr
      β = B[p, j]
      @unroll for i = 1:mr_vl
        c[i, j] = muladd(a[i], β, c[i, j])
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
