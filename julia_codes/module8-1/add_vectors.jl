using CUDA

# Loop based function
function addvecs!(c, a, b)
  N = length(a)
  @inbounds for i = 1:N
    c[i] = a[i] + b[i]
  end
end

# Loop based fake "GPU" kernel
function fake_knl_addvecs!(c, a, b, numblocks, bdim)
  N = length(a)
  # loop over the "blocks"
  @inbounds for bidx = 1:numblocks
    # loop over the "threads"
    for tidx = 1:bdim
      i = (bidx - 1) * bdim + tidx
      if i <= N
        c[i] = a[i] + b[i]
      end
    end
  end
end

# Real GPU kernel
function knl_addvecs!(c, a, b)

  N = length(a)

  tidx = threadIdx().x # get the thread ID
  bidx = blockIdx().x  # get the block ID
  bdim = blockDim().x  # how many threads in each block

  # figure out which index we should handle
  i = (bidx - 1) * bdim + tidx
  @inbounds if i <= N
    c[i] = a[i] + b[i]
  end

  #Kernel must return nothing
  nothing
end

let
  N = 100000000
  a = rand(N)
  b = rand(N)

  #
  # Reference Implementation
  #
  c0 = similar(a)
  c0 .= a .+ b
  @time c0 .= a .+ b

  #
  # Simple for loop implementation
  #
  c1 = similar(c0)
  addvecs!(c1, a, b)
  @time addvecs!(c1, a, b)
  @assert c0 ≈ c1

  #
  # fake GPU kernel
  #
  c2 = similar(c0)
  numthreads = 256
  numblocks = div(N + numthreads - 1, numthreads)
  fake_knl_addvecs!(c2, a, b, numblocks, numthreads)
  @time fake_knl_addvecs!(c2, a, b, numblocks, numthreads)
  @assert c0 ≈ c2

  #
  # Real GPU call
  #
  d_a = CuArray(a) # Copy data to GPU
  d_b = CuArray(b) # Copy data to GPU
  d_c3 = CuArray{Float64}(undef, N) # Allocate array on GPU
  # launch GPU kernel
  @cuda threads=numthreads blocks=numblocks knl_addvecs!(d_c3, d_a, d_b)
  # Barrier before timing
  synchronize()
  @time begin
    @cuda threads=numthreads blocks=numblocks knl_addvecs!(d_c3, d_a, d_b)
    # Barrier before completing
    synchronize()
  end

  # Copy back to the CPU
  c3 = Array(d_c3)
  @assert c0 ≈ c3

  #
  # Try Julia's native CUDA version
  #
  d_c3 .= d_a .+ d_b
  synchronize()
  @time begin
    d_c3 .= d_a .+ d_b
    synchronize()
  end
end
nothing
