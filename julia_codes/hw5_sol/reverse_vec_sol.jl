using CUDA

# Loop based function
function reversevec!(a, b)
  N = length(a)
  for i = 1:N
    a[i] = b[N - i + 1]
  end
end

# Loop based fake "GPU" kernel
function fake_knl_reversevecs!(a, b, numblocks, bdim)
  N = length(a)
  # loop over the "blocks"
  @inbounds for bidx = 1:numblocks
    # loop over the "threads"
    for tidx = 1:bdim
      i = (bidx - 1) * bdim + tidx
      if i <= N
        a[i] = b[N - i + 1]
      end
    end
  end
end

# Real GPU kernel
function knl_reversevecs!(a, b)

  N = length(a)

  tidx = threadIdx().x # get the thread ID
  bidx = blockIdx().x  # get the block ID
  bdim = blockDim().x  # how many threads in each block

  # figure out which index we should handle
  i = (bidx - 1) * bdim + tidx
  @inbounds if i <= N
    a[i] = b[N - i + 1]
  end

  #Kernel must return nothing
  nothing
end

# Real GPU kernel
function knl_reversevecs_inplace_bad!(a)

  N = length(a)

  tidx = threadIdx().x # get the thread ID
  bidx = blockIdx().x  # get the block ID
  bdim = blockDim().x  # how many threads in each block

  # figure out which index we should handle
  i = (bidx - 1) * bdim + tidx
  @inbounds if i <= N
    a[i] = a[N - i + 1]
  end

  #Kernel must return nothing
  nothing
end

# Real GPU kernel
function knl_reversevecs_inplace!(a)

  N = length(a)

  tidx = threadIdx().x # get the thread ID
  bidx = blockIdx().x  # get the block ID
  bdim = blockDim().x  # how many threads in each block

  # figure out which index we should handle
  i = (bidx - 1) * bdim + tidx
  @inbounds if i <= div(N, 2)
    a1 = a[i]
    a2 = a[N - i + 1]
    a[i] = a2
    a[N - i + 1] = a1
  end

  #Kernel must return nothing
  nothing
end

### Start of testing/driver code
let
  T = Float64
  N = 10000000
  b = rand(T, N)
  a_ref = b[end:-1:1]

  #
  ### Simple reference for loop CPU implementation call
  #
  print("First version: reference loop-based double array reverse \n")
  a1 = similar(a_ref)
  # "Warm-up" call
  reversevec!(a1, b)
  # Timing call
  @time reversevec!(a1, b)
  @assert a1 == a_ref

  #
  ### Fake GPU kernel call
  #
  print("Second version: fake GPU kernel \n")
  a2 = similar(a_ref)
  numthreads = 256
  numblocks = div(N + numthreads - 1, numthreads)
  # "Warm-up" call
  fake_knl_reversevecs!(a2, b, numblocks, numthreads)
  # Timing call
  @time fake_knl_reversevecs!(a2, b, numblocks, numthreads)
  @assert a_ref ≈ a2

  #
  ### Double array reverse call
  #
  print("Third version: real double-array reverse GPU kernel \n")
  d_b = CuArray(b) # Copy data to GPU
  d_a3 = CuArray{T}(undef, N) # Allocate array on GPU
  # "Warm-up" call
  # Launch GPU kernel
  @cuda threads=numthreads blocks=numblocks knl_reversevecs!(d_a3, d_b)
  # Barrier before timing
  synchronize()
  @time begin
    # Timing call
    @cuda threads=numthreads blocks=numblocks knl_reversevecs!(d_a3, d_b)
    # Barrier before completing
    synchronize()
  end

  # Copy back to the CPU
  a3 = Array(d_a3)
  # Check correctness of result
  @assert a_ref ≈ a3

  #
  ### Inplace reverse GPU kernel (BAD) call
  #
  print("Fourth version: inplace GPU kernel (BAD) \n")
  d_a4 = CuArray(b)
  # "Warm-up" call
  # Launch GPU kernel
  @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace_bad!(d_a4)
  # Barrier before timing
  synchronize()
  d_a4 = CuArray(b) # need to reinitialize because the previous call to knl_reversevecs_inplace_bad! changed its content
  @time begin
    # Timing call
    @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace_bad!(d_a4)
    # Barrier before completing
    synchronize()
  end

  # Copy back to the CPU
  a4 = Array(d_a4)
  # Check correctness of result
  # @assert a_ref ≈ a4 # If you try to uncomment this line, it will break.... why? The knl_reversevecs_inplace_bad! kernel does not guarantee correct results, since to swap any two entries/values you always need at least a third temporary entry/value.

  #
  ### Inplace correct reverse GPU kernel call
  #
  print("Fifth version: correct inplace GPU kernel \n")
  d_a5 = CuArray(b)
  # "Warm-up" call
  # Launch GPU kernel
  @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace!(d_a5)
  # Barrier before timing
  synchronize()
  d_a5 = CuArray(b) # need to reinitialize because the previous call to knl_reversevecs_inplace_bad! changed its content
  @time begin
    # Timing call
    @cuda threads=numthreads blocks=numblocks knl_reversevecs_inplace!(d_a5)
    # Barrier before completing
    synchronize()
  end

  # Copy back to the CPU
  a5 = Array(d_a5)
  # Check correctness of result
  @assert a_ref ≈ a5 # This time the assert check passes, since the correct version of the kernel guarantees that the array is flipped properly

end
nothing
