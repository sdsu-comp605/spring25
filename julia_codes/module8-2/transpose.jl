#=
This code has been written using three codes as influence: one from Lucas Wilcox, one (the transpose code) from Andreas Kloeckner available at

https://github.com/hpc12/lec4-demo

and a code by Mark Harris found at

http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/

and

https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu

with the license

Copyright 2012 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=#

using CUDA
using Printf
using StaticArrays
using KernelAbstractions.Extras: @unroll
CUDA.allowscalar(false) # disallow scalar indexing to make sure loops are executed on the GPU

function copy_naive!(b, a, ::Val{TILE_DIM}) where TILE_DIM
  N = size(a, 1)
  i = (blockIdx().x-1) * TILE_DIM + threadIdx().x
  j = (blockIdx().y-1) * TILE_DIM + threadIdx().y

  if i <= N && j <= N
    @inbounds b[i, j] = a[i, j]
  end
  nothing
end

function copy_prefetch_tile!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}
  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  i  = (blockIdx().x - 1) * TILE_DIM + tidx
  j0 = (blockIdx().y - 1) * TILE_DIM + tidy
  NE_ROW = div(TILE_DIM, BLOCK_ROWS)

  tile = MArray{Tuple{NE_ROW}, eltype(a)}(undef) # An array whose size is known at compile time that is mutable

  @unroll for k = 0:NE_ROW-1
    j = j0 + k * BLOCK_ROWS
    if i <= N && j <= N
      @inbounds tile[k+1] = a[i, j]
    end
  end
  @unroll for k = 0:NE_ROW-1
    j = j0 + k * BLOCK_ROWS
    if i <= N && j <= N
      @inbounds b[i, j] = tile[k+1]
    end
  end

  nothing
end

function copy_shared!(b, a, ::Val{TILE_DIM}) where TILE_DIM
  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  i = (blockIdx().x-1) * TILE_DIM + tidx
  j = (blockIdx().y-1) * TILE_DIM + tidy
  tile = @cuStaticSharedMem(eltype(a), (TILE_DIM, TILE_DIM))

  if i <= N && j <= N
    @inbounds tile[tidx, tidy] = a[i, j]
  end

  sync_threads()

  if i <= N && j <= N
    @inbounds b[i, j] = tile[tidx, tidy]
  end

  nothing
end

function copy_tiled!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}
  N = size(a, 1)
  i = (blockIdx().x-1) * TILE_DIM + threadIdx().x
  j = (blockIdx().y-1) * TILE_DIM + threadIdx().y

  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    if i <= N && (j+k) <= N
      @inbounds b[i, j+k] = a[i, j+k]
    end
  end
end

function copy_tiled_shared!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}
  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  i = (blockIdx().x-1) * TILE_DIM + tidx
  j = (blockIdx().y-1) * TILE_DIM + tidy

  tile = @cuStaticSharedMem(eltype(a), (TILE_DIM, TILE_DIM))

  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    if i <= N && (j+k) <= N
      @inbounds tile[tidx, tidy+k] = a[i, j+k]
    end
  end

  sync_threads()

  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    if i <= N && (j+k) <= N
      @inbounds b[i, j+k] = tile[tidx, tidy+k]
    end
  end

  nothing
end

function transpose_naive!(b, a, ::Val{TILE_DIM}) where TILE_DIM
  N = size(a, 1)
  i = (blockIdx().x-1) * TILE_DIM + threadIdx().x
  j = (blockIdx().y-1) * TILE_DIM + threadIdx().y

  if i <= N && j <= N
    @inbounds b[i, j] = a[j, i]
  end

  nothing
end

function transpose_multiple!(b, a, ::Val{TILE_DIM}, ::Val{STRIDE}) where {TILE_DIM, STRIDE}
  N = size(a, 1)
  NUM_ELEM = div(TILE_DIM, STRIDE)

  i0 = (blockIdx().x-1) * TILE_DIM + threadIdx().x
  j  = (blockIdx().y-1) * TILE_DIM + threadIdx().y

  tile = MArray{Tuple{NUM_ELEM}, eltype(a)}(undef) # An array whose size is known at compile time that is mutable

  if j <= N
    @unroll for k = 0:NUM_ELEM-1
      i = i0 + STRIDE*k
      if i <= N
        @inbounds tile[k+1] = a[j, i]
      end
    end
    @unroll for k = 0:NUM_ELEM-1
      i = i0 + STRIDE*k
      if i <= N
        @inbounds b[i, j] = tile[k+1]
      end
    end
  end

  nothing
end

function transpose_shared!(b, a, ::Val{TILE_DIM}) where TILE_DIM
  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  bidx, bidy = blockIdx().x, blockIdx().y
  i = (bidx-1) * TILE_DIM + tidx
  j = (bidy-1) * TILE_DIM + tidy
  tile = @cuStaticSharedMem(eltype(a), (TILE_DIM, TILE_DIM))

  if i <= N && j <= N
    @inbounds tile[tidx, tidy] = a[i, j]
  end

  sync_threads()

  i = (bidy-1) * TILE_DIM + tidx
  j = (bidx-1) * TILE_DIM + tidy

  if i <= N && j <= N
    @inbounds b[i, j] = tile[tidy, tidx]
  end

  nothing
end

function transpose_tiled!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}
  N = size(a, 1)
  i = (blockIdx().x-1) * TILE_DIM + threadIdx().x
  j = (blockIdx().y-1) * TILE_DIM + threadIdx().y

  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    if i <= N && (j+k) <= N
      @inbounds b[i, j+k] = a[j+k, i]
    end
  end
end


function transpose_tiled_prefetch!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}

  NE_ROW = div(TILE_DIM, BLOCK_ROWS)
  tile = MArray{Tuple{NE_ROW}, eltype(a)}(undef) # An array whose size is known at compile time that is mutable

  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  i  = (blockIdx().x - 1) * TILE_DIM + tidx
  j0 = (blockIdx().y - 1) * TILE_DIM + tidy

  if i <= N
    @unroll for k = 0:NE_ROW-1
      j = j0 + k * BLOCK_ROWS
      if j <= N
        @inbounds tile[k+1] = a[j, i]
      end
    end
    @unroll for k = 0:NE_ROW-1
      j = j0 + k * BLOCK_ROWS
      if j <= N
        @inbounds b[i, j] = tile[k+1]
      end
    end
  end
end

function transpose_tiled_shared!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}
  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  i = (blockIdx().x-1) * TILE_DIM + tidx
  j = (blockIdx().y-1) * TILE_DIM + tidy

  tile = @cuStaticSharedMem(eltype(a), (TILE_DIM, TILE_DIM))

  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    if i <= N && (j+k) <= N
      @inbounds tile[tidy+k, tidx] = a[i, j+k]
    end
  end

  sync_threads()

  i = (blockIdx().y-1) * TILE_DIM + tidx
  j = (blockIdx().x-1) * TILE_DIM + tidy
  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    if i <= N && (j+k) <= N
      @inbounds b[i, j+k] = tile[tidx, tidy+k]
    end
  end

  nothing
end

function transpose_tiled_shared_noconflicts!(b, a, ::Val{TILE_DIM}, ::Val{BLOCK_ROWS}) where {TILE_DIM, BLOCK_ROWS}
  N = size(a, 1)
  tidx, tidy = threadIdx().x, threadIdx().y
  bidx, bidy = blockIdx().x, blockIdx().y
  i = (bidx-1) * TILE_DIM + tidx
  j = (bidy-1) * TILE_DIM + tidy

  tile = @cuStaticSharedMem(eltype(a), (TILE_DIM+1, TILE_DIM))

  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    # if i <= N && (j+k) <= N
      @inbounds tile[tidy+k, tidx] = a[i, j+k]
    # end
  end

  sync_threads()

  i = (bidy-1) * TILE_DIM + tidx
  j = (bidx-1) * TILE_DIM + tidy
  @unroll for k = 0:BLOCK_ROWS:TILE_DIM-1
    # if i <= N && (j+k) <= N
      @inbounds b[i, j+k] = tile[tidx, tidy+k]
    # end
  end

  nothing
end

function tester(fun!, b_ref, d_b, d_a, blockdim, griddim, num_reps, args...)
  @printf("%40s", fun!)
  @cuda threads=blockdim blocks=griddim fun!(d_b, d_a, args...)
  fill!(d_b, 0)

  synchronize()
  t1 = time_ns()
  for i = 1:num_reps
    @cuda threads=blockdim blocks=griddim fun!(d_b, d_a, args...)
  end
  synchronize()
  t2 = time_ns()

  @assert b_ref == Array(d_b)
  nanoseconds = (t2-t1)

  memsize = length(b_ref) * sizeof(eltype(b_ref))
  @printf("%20.2f\n", 2 * memsize * num_reps / nanoseconds)
end

function main(;N=1024, DFloat = Float32, num_reps=1000,
              TILE_DIM = 32, BLOCK_ROWS = 8)

  memsize = N * N * sizeof(DFloat)

  griddim = (div(N+TILE_DIM-1, TILE_DIM), div(N+TILE_DIM-1, TILE_DIM))
  blockdim = (TILE_DIM, BLOCK_ROWS)

  println(device()) # Calling this function at the start of a session will make sure CUDA is initialized (i.e., a primary context will be created and activated) and here we print the device ID.

  print("Matrix Size: $N $N, ")
  print("Block size: $TILE_DIM $BLOCK_ROWS, ")
  println("Tile size: $TILE_DIM $TILE_DIM")
  print("griddim: $(griddim[1]) $(griddim[2]), ")
  println("blockdim: $(blockdim[1]) $(blockdim[2])")
  println("memory per matrix: $(memsize/1024^3) GiB")

  a = rand(DFloat, N, N)
  b = similar(a)
  d_a = CuArray(a)
  d_b = CuArray{DFloat}(undef, N, N)

  @printf("%40s%25s\n", "Routine", "Bandwidth (GB/s)")

  ##############
  # NAIVE COPY #
  ##############
  tester(copy_naive!, a, d_b, d_a, (TILE_DIM, TILE_DIM), griddim, num_reps,
         Val(TILE_DIM))

  ###############
  # SHARED COPY #
  ###############
  tester(copy_prefetch_tile!, a, d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim,
         num_reps, Val(TILE_DIM), Val(BLOCK_ROWS))

  ###############
  # SHARED COPY #
  ###############
  tester(copy_shared!, a, d_b, d_a, (TILE_DIM, TILE_DIM), griddim, num_reps,
         Val(TILE_DIM))

  #############n#
  # TILED COPY #
  ##############
  tester(copy_tiled!, a, d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim, num_reps,
         Val(TILE_DIM), Val(BLOCK_ROWS))

  #####################
  # TILED SHARED COPY #
  #####################
  tester(copy_tiled_shared!, a, d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim, num_reps,
         Val(TILE_DIM), Val(BLOCK_ROWS))

  ###################
  # NAIVE TRANSPOSE #
  ###################
  tester(transpose_naive!, a', d_b, d_a, (TILE_DIM, TILE_DIM), griddim, num_reps,
         Val(TILE_DIM))

  ###################
  # NAIVE TRANSPOSE #
  ###################
  STRIDE = div(TILE_DIM, 4)
  tester(transpose_multiple!, a', d_b, d_a, (STRIDE, TILE_DIM), griddim,
         num_reps, Val(TILE_DIM), Val(STRIDE))

  ###############
  # SHARED COPY #
  ###############
  tester(transpose_tiled_prefetch!, a', d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim,
         num_reps, Val(TILE_DIM), Val(BLOCK_ROWS))


  ####################
  # SHARED TRANSPOSE #
  ####################
  tester(transpose_shared!, a', d_b, d_a, (TILE_DIM, TILE_DIM), griddim, num_reps,
         Val(TILE_DIM))

  ###################
  # TILED TRANSPOSE #
  ###################
  tester(transpose_tiled!, a', d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim, num_reps,
         Val(TILE_DIM), Val(BLOCK_ROWS))

  ###########################
  # TILED, SHARED TRANSPOSE #
  ###########################
  tester(transpose_tiled_shared!, a', d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim, num_reps,
         Val(TILE_DIM), Val(BLOCK_ROWS))

  ###########################
  # TILED, SHARED TRANSPOSE #
  ###########################
  tester(transpose_tiled_shared_noconflicts!, a', d_b, d_a, (TILE_DIM, BLOCK_ROWS), griddim, num_reps,
         Val(TILE_DIM), Val(BLOCK_ROWS))

  nothing
end

main()
