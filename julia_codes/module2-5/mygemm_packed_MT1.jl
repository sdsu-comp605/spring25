# The realization to use a linear vector for packing is due to Maciej
# Waruszewski
#
#     https://github.com/mwarusz/MyJuliaGEMM
#

using KernelAbstractions.Extras: @unroll
using SIMD
using StaticArrays

struct Sizes{mc, nc, kc, mr, nr, vl} end
Base.elsize(::StaticArray{Tp, T}) where {Tp, T} = sizeof(T)

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

function mygemm_packed_MT1!(C, A, B, sizes::Sizes)
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
  @assert size(A) == (n, k)
  @assert size(B, 2) == m

  # We assume that the matrices are even divisible by `nr` and `mr`
  @assert mod(m, mr) == 0
  @assert mod(n, nr) == 0

  # We assume that the blocking perfectly partitions the matrix
  @assert mod(mc, mr) == 0
  @assert mod(nc, nr) == 0

  # Microvector are perfectly by the chosen vector unit
  @assert mod(mr, vl) == 0

  T = eltype(C)
  # A_pack = MArray{Tuple{mc * kc}, T}(undef)
  # B_pack = MArray{Tuple{kc * nc}, T}(undef)
  A_pack = Array{T}(undef, mc * kc)
  B_pack = Array{T}(undef, kc * nc)

  mygemm_JPI_JI_packed_MT1!(C, A, B, A_pack, B_pack, sizes)
end

function mygemm_JPI_JI_packed_MT1!(C, A, B, A_pack, B_pack, sizes::Sizes)
  n = size(C, 2)

  nc = sizes.nc
  @inbounds for j1 = 1:nc:n
    j2 = min(n, j1 + nc - 1)
    C_J = @view C[:, j1:j2]
    B_J = @view B[:, j1:j2]
    mygemm_PI_JI_packed_MT1!(C_J, A, B_J, A_pack, B_pack, sizes)
  end
end

@inline function packB!(B_pack, B, sizes::Sizes)
  k, n = size(B)
  nr = sizes.nr
  # Loop through B in access order to store in packed array
  liner_ix = 1
  @inbounds for x = 1:nr:n        # which tile * nr
    @unroll 4 for p in 1:k        # which row
      @simd ivdep for y = 0:nr-1  # micro vector element
        B_pack[liner_ix] = B[p, x + y]
        liner_ix += 1
      end
    end
  end
end

@inline function mygemm_PI_JI_packed_MT1!(C, A, B, A_pack, B_pack, sizes::Sizes)
  k = size(A, 2)

  kc = sizes.kc
  mr = sizes.mr
  nr = sizes.nr
  @inbounds for p1 = 1:kc:k
    p2 = min(k, p1 + kc - 1)
    A_P = @view A[:, p1:p2]
    B_P = @view B[p1:p2, :]
    real_kc = (p2 - p1 + 1)
    kc_nr = real_kc * nr
    kc_mr = real_kc * mr
    packB!(B_pack, B_P, sizes)
    mygemm_I_JI_packed_MT1!(C, A_P, A_pack, B_pack, kc_nr, kc_mr, sizes)
  end
end

@inline function packA!(A_pack, A, sizes::Sizes)
  m, k = size(A)
  mr = sizes.mr
  # Loop through B in access order to store in packed array
  liner_ix = 1
  @inbounds for x = 1:mr:m        # which tile * mr
    @unroll 4 for p in 1:k        # which row
      @simd ivdep for y = 0:mr-1  # micro vector element
        A_pack[liner_ix] = A[x + y, p]
        liner_ix += 1
      end
    end
  end
end

@inline function mygemm_I_JI_packed_MT1!(C, A, A_pack, B_pack, kc_nr, kc_mr, sizes::Sizes)
  m = size(C, 1)

  mc = sizes.mc
  @inbounds for i1 = 1:mc:m
    i2 = min(m, i1 + mc - 1)
    C_I = @view C[i1:i2, :]
    A_I = @view A[i1:i2, :]
    packA!(A_pack, A_I, sizes)
    mygemm_JI_packed_MT1!(C_I, A_pack, B_pack, kc_nr, kc_mr, sizes)
  end
end

@inline function mygemm_JI_packed_MT1!(C, A_pack, B_pack, kc_nr, kc_mr, sizes::Sizes)
  n = size(C, 2)

  nr = sizes.nr
  @inbounds for (t, j1) = enumerate(1:nr:n)
    j2 = min(n, j1 + nr - 1)
    C_J = @view C[:, j1:j2]
    tile_rng = (1 + kc_nr * (t - 1)) : (t * kc_nr)
    B_pack_J = @view B_pack[tile_rng]
    mygemm_I_packed_MT1!(C_J, A_pack, B_pack_J, kc_mr, sizes)
  end
end

@inline function mygemm_I_packed_MT1!(C, A_pack, B_pack, kc_mr, sizes::Sizes)
  m = size(C, 1)

  mr = sizes.mr
  @inbounds Threads.@threads for t = 1:cld(m, mr)
    i1 = (t - 1) * mr + 1
    i2 = min(m, i1 + mr - 1)
    C_I = @view C[i1:i2, :]
    tile_rng = (1 + kc_mr * (t - 1)) : (t * kc_mr)
    A_pack_I = @view A_pack[tile_rng]
    microknl_packed_MT1!(C_I, A_pack_I, B_pack, sizes)
  end
end

@inline function microknl_packed_MT1!(C, A, B, sizes::Sizes)
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
  pend = div(size(B, 1), nr)

  # storage for micro-vectors of A
  a = MVector{mr_vl, VecT}(undef)

  # Do the rank-one updates for each p
  # (Unroll means unroll the loop by the factor of 4)
  @inbounds @unroll 4 for p = 1:pend
    # load pieces of `a`
    @unroll for i = 1:mr_vl
      offset = ((p - 1) * mr + (i - 1) * vl) * sizeof(T)
      a[i] = vload(VecT, pointer(A) + offset)
    end
    # do outer product for each piece
    @unroll for j = 1:nr
      β = B[j + nr * (p - 1)]
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
