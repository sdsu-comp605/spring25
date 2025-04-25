#=
This is a Julia implementation of the reduction kernels described by Mark
Harris in

    http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

Acknowledgements to Jeremy Kozdon (NPS)
=#

using CUDA
using Printf
CUDA.allowscalar(false)

function reduction_v0!(d_x, numruns)
  val   = zero(eltype(d_x))
  for n = 1:numruns
    val = sum(d_x) # use Julia's built-in sum
  end
  val
end

"""
   reduction_knl_v1(x, y; op=+, ldim::Val{LDIM}=Val(256))

Basic interleaved memory access version where each `LDIM` chunk of values from `x`
are reduced using `op` (+ by default) to single values which are stored in the first
`gridDim().x` values of `y`

Problems:
  - Modulo is slow on the GPU >> Remove by calculating
  - threads in a warp are divergent (for the if statement) >> shift threads as algorithm progresses
"""
function knl_reduction_v1!(y, x, N, op, ::Val{LDIM}) where LDIM
  tid = threadIdx().x
  bid = blockIdx().x
  gid = tid + (bid - 1) * LDIM # global thread index
  l_x = @cuStaticSharedMem(eltype(x), LDIM) # shared memory allocation

  @inbounds begin
    # set the local/shared memory array
    if gid <= N
      l_x[tid] = x[gid]
    else
      l_x[tid] = 0
    end

    sync_threads()

    s = 1
    while s < LDIM
      # I'm still active if my thread ID (minus 1) is divisible by 2*s
      if ((tid-1) % (2 * s) == 0)
        # combine my value to my (interleaved/strided) neighbors value
        l_x[tid] = op(l_x[tid], l_x[tid + s])
      end
      s *= 2
      sync_threads()
    end

    # Thread 1 is the only remaining "real" thread which will carry the result
    tid == 1 && (y[bid] = l_x[tid])
  end

  nothing
end

"""
   reduction_v1!(d_x, numruns; op=+, ldim=256)

Kernel launcher function where we set execution configuration for the knl_reduction_v1! kernel
"""
function reduction_v1!(d_x, numruns; op=+, ldim=256)
  N = length(d_x)

  nblocks = div(N + ldim - 1, ldim)
  d_r1 = CuArray{eltype(d_x)}(undef, nblocks)
  d_r2 = CuArray{eltype(d_x)}(undef, nblocks)

  val = zeros(eltype(d_x), 1)
  for n = 1:numruns
    Nelem = N
    nblocks = div(Nelem + ldim - 1, ldim)
    @cuda threads=ldim blocks=nblocks knl_reduction_v1!(d_r1, d_x, Nelem, op, Val(ldim))
    while nblocks > 1
      Nelem = nblocks
      nblocks = div(Nelem + ldim - 1, ldim)
      @cuda threads=ldim blocks=nblocks knl_reduction_v1!(d_r2, d_r1, Nelem, op, Val(ldim))
      d_r1, d_r2 = d_r2, d_r1
    end
    copyto!(val, 1, d_r1, 1, 1)
  end
  val[1]
end


"""
   reduction_knl_v2(x, y; op=+, ldim::Val{LDIM}=Val(256))

Improved strided memory access version where each `LDIM` chunk of values from `x`
are reduced using `op` to single values which are stored in the first
`gridDim().x` values of `y`

Problems:
  - Memory access is still strided, thus bank conflicts >> sequential access
"""
function knl_reduction_v2!(y, x, N, op, ::Val{LDIM}) where LDIM
  tid = threadIdx().x
  bid = blockIdx().x
  gid = tid + (bid - 1) * LDIM
  l_x = @cuStaticSharedMem(eltype(x), LDIM)

  @inbounds begin
    if gid <= N
      l_x[tid] = x[gid]
    else
      l_x[tid] = 0
    end

    sync_threads()

    s = 1
    while s < LDIM
      # figure out whether I'm in the active block
      sid = 2 * (tid-1) * s + 1
      if sid + s <= LDIM
        # combine my value to my (strided) neighbors value
        l_x[sid] = op(l_x[sid], l_x[sid+s])
      end
      s *= 2
      sync_threads()
    end

    # Thread 1 is the only remaining "real" thread
    tid == 1 && (y[bid] = l_x[tid])
  end

  nothing
end

"""
  reduction_v2!(d_x, numruns; op=+, ldim=256)

Kernel launcher function where we set execution configuration for the knl_reduction_v2! kernel
"""
function reduction_v2!(d_x, numruns; op=+, ldim=256)
  N = length(d_x)

  nblocks = div(N + ldim - 1, ldim)
  d_r1 = CuArray{eltype(d_x)}(undef, nblocks)
  d_r2 = CuArray{eltype(d_x)}(undef, nblocks)

  val = zeros(eltype(d_x), 1)
  for n = 1:numruns
    Nelem = N
    nblocks = div(Nelem + ldim - 1, ldim)
    @cuda threads=ldim blocks=nblocks knl_reduction_v2!(d_r1, d_x, Nelem, op, Val(ldim))
    while nblocks > 1
      Nelem = nblocks
      nblocks = div(Nelem + ldim - 1, ldim)
      @cuda threads=ldim blocks=nblocks knl_reduction_v2!(d_r2, d_r1, Nelem, op, Val(ldim))
      d_r1, d_r2 = d_r2, d_r1
    end
    copyto!(val, 1, d_r1, 1, 1)
  end
  val[1]
end

"""
   reduction_knl_v3(x, y; op=+, ldim::Val{LDIM}=Val(256))

Basic sequential memory access version where each `LDIM` chunk of values from
`x` are reduced using `op` to single values which are stored in the first
`gridDim().x` values of `y`

Problems:
  - Some threads do very little work (idle threads) >> have each thread add in a few values
"""
function knl_reduction_v3!(y, x, N, op, ::Val{LDIM}) where LDIM
  tid = threadIdx().x
  bid = blockIdx().x
  gid = tid + (bid - 1) * LDIM
  l_x = @cuStaticSharedMem(eltype(x), LDIM)

  @inbounds begin
    if gid <= N
      l_x[tid] = x[gid]
    else
      l_x[tid] = 0
    end

    sync_threads()

    # bit right shift to divide by 2
    s = LDIM >> 1
    while s > 0
      # I'm still active if my thread ID is less than
      if tid <= s
        # combine my value to my (strided) neighbors value
        l_x[tid] = op(l_x[tid], l_x[tid + s])
      end
      # bit right shift to divide by 2
      s = s >> 1
      sync_threads()
    end

    # Thread 1 is the only remaining "real" thread
    tid == 1 && (y[bid] = l_x[tid])
  end

  nothing
end

"""
  reduction_v3!(d_x, numruns; op=+, ldim=256)

Kernel launcher function where we set execution configuration for the knl_reduction_v3! kernel
"""
function reduction_v3!(d_x, numruns; op=+, ldim=256)
  N = length(d_x)

  nblocks = div(N + ldim - 1, ldim)
  d_r1 = CuArray{eltype(d_x)}(undef, nblocks)
  d_r2 = CuArray{eltype(d_x)}(undef, nblocks)

  val = zeros(eltype(d_x), 1)
  for n = 1:numruns
    Nelem = N
    nblocks = div(Nelem + ldim - 1, ldim)
    @cuda threads=ldim blocks=nblocks knl_reduction_v3!(d_r1, d_x, Nelem, op, Val(ldim))
    while nblocks > 1
      Nelem = nblocks
      nblocks = div(Nelem + ldim - 1, ldim)
      @cuda threads=ldim blocks=nblocks knl_reduction_v3!(d_r2, d_r1, Nelem, op, Val(ldim))
      d_r1, d_r2 = d_r2, d_r1
    end
    copyto!(val, 1, d_r1, 1, 1)
  end
  val[1]
end

"""
   reduction_knl_v4(x, y; op=+, ldim::Val{LDIM}=Val(256))

Add a few values before switching to shared memory with sequential memory
access version where each `LDIM` chunk of values from `x` are reduced using
`op` to single values which are stored in the first `gridDim().x` values of `y`
"""
function knl_reduction_v4!(y, x, N, op, ::Val{LDIM},
                           ::Val{OVERLAP}) where {LDIM, OVERLAP}
  tid = threadIdx().x
  bid = blockIdx().x
  gid = tid + (bid - 1) * LDIM
  gsz = LDIM * gridDim().x # total number of threads

  l_x = @cuStaticSharedMem(eltype(x), LDIM)

  @inbounds begin
    # Have each thread initially load and add OVERLAP values
    p_x = zero(eltype(x))
    for n = 1:OVERLAP
      if gid + (n-1) * gsz <= N
        p_x += x[gid + (n-1)*gsz]
      end
    end

    l_x[tid] = p_x

    sync_threads()

    # bit right shift to divide by 2
    s = LDIM >> 1
    while s > 0
      # I'm still active if my thread ID is less than
      if tid <= s
        # combine my value to my (strided) neighbors value
        l_x[tid] = op(l_x[tid], l_x[tid + s])
      end
      s = s >> 1 # bit right shift to divide by 2 again
      sync_threads()
    end

    # Thread 1 is the only remaining "real" thread
    tid == 1 && (y[bid] = l_x[tid])
  end

  nothing
end

"""
  reduction_v4!(d_x, numruns; op=+, ldim=256, overlap=16)

Kernel launcher function where we set execution configuration for the knl_reduction_v4! kernel
"""
function reduction_v4!(d_x, numruns; op=+, ldim=256, overlap=16)
  N = length(d_x)

  nblocks = div(N + ldim*overlap - 1, ldim*overlap)
  d_r1 = CuArray{eltype(d_x)}(undef, nblocks)
  d_r2 = CuArray{eltype(d_x)}(undef, nblocks)

  val = zeros(eltype(d_x), 1)
  for n = 1:numruns
    Nelem = N
    nblocks = div(Nelem + ldim*overlap - 1, ldim*overlap)
    @cuda threads=ldim blocks=nblocks knl_reduction_v4!(d_r1, d_x, Nelem, op,
                                                        Val(ldim),
                                                        Val(overlap))
    while nblocks > 1
      Nelem = nblocks
      nblocks = div(Nelem + ldim*overlap - 1, ldim*overlap)
      @cuda threads=ldim blocks=nblocks knl_reduction_v4!(d_r2, d_r1, Nelem,
                                                          op, Val(ldim),
                                                          Val(overlap))
      d_r1, d_r2 = d_r2, d_r1
    end
    copyto!(val, 1, d_r1, 1, 1)
  end
  val[1]
end

## main function
function main(; N=1024^2, numruns = 10, DFloat = Float64, ldim = 256, overlap=16)

  x = collect(DFloat, 1:N)
  d_x = CuArray(x)
  exact = div(N * (N+1), 2)

  memsize = N * sizeof(DFloat)
  @printf("number of values: %d\n", N)
  @printf("number of run   : %d\n", numruns)
  @printf("memsize         : %e GB\n", memsize * 1e-9)
  @printf("device            : %s\n", device())
  @printf("%30s%20s%20s%10s\n", "Routine", "Total Time (s)", "Bandwidth (GB/s)", "check")

  # Version 0
  computed = reduction_v0!(d_x, 1)
  synchronize()
  t1 = time_ns()
  computed = reduction_v0!(d_x, numruns)
  synchronize()
  t2 = time_ns()

  nanoseconds = t2 - t1
  @printf("%30s", "CuArrays reduction")
  @printf("%16.2e", nanoseconds*1e-9)
  @printf("%18.2f", memsize * numruns / nanoseconds)
  @printf("%16s\n", computed == exact)

  # Version 1
  computed = reduction_v1!(d_x, 1, ldim=ldim)
  synchronize()
  t1 = time_ns()
  computed = reduction_v1!(d_x, numruns, ldim=ldim)
  synchronize()
  t2 = time_ns()

  nanoseconds = t2 - t1
  @printf("%30s", "strided memory access")
  @printf("%16.2e", nanoseconds*1e-9)
  @printf("%18.2f", memsize * numruns / nanoseconds)
  @printf("%16s\n", computed == exact)

  # Version 2
  computed = reduction_v2!(d_x, 1, ldim=ldim)
  synchronize()
  t1 = time_ns()
  computed = reduction_v2!(d_x, numruns, ldim=ldim)
  synchronize()
  t2 = time_ns()

  nanoseconds = t2 - t1
  @printf("%30s", "better strided memory access")
  @printf("%16.2e", nanoseconds*1e-9)
  @printf("%18.2f", memsize * numruns / nanoseconds)
  @printf("%16s\n", computed == exact)

  # Version 3
  computed = reduction_v3!(d_x, 1, ldim=ldim)
  synchronize()
  t1 = time_ns()
  computed = reduction_v3!(d_x, numruns, ldim=ldim)
  synchronize()
  t2 = time_ns()

  nanoseconds = t2 - t1
  @printf("%30s", "sequential memory access")
  @printf("%16.2e", nanoseconds*1e-9)
  @printf("%18.2f", memsize * numruns / nanoseconds)
  @printf("%16s\n", computed == exact)

  # Version 4
  computed = reduction_v4!(d_x, 1, ldim=ldim, overlap=overlap)
  synchronize()
  t1 = time_ns()
  computed = reduction_v4!(d_x, numruns, ldim=ldim, overlap=overlap)
  synchronize()
  t2 = time_ns()

  nanoseconds = t2 - t1
  @printf("%30s", "overlap initial")
  @printf("%16.2e", nanoseconds*1e-9)
  @printf("%18.2f", memsize * numruns / nanoseconds)
  @printf("%16s\n", computed == exact)
end

main()
