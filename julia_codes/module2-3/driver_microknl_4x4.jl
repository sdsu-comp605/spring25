# What modules / packages do we depend on
using Random
using LinearAlgebra
using Printf

# To ensure repeatability
Random.seed!(777)

# Don't let BLAS use lots of threads (since we are not multi-threaded yet!)
BLAS.set_num_threads(1)

include("mygemm_microknl_4x4.jl")

# C := α * A * B + β * C
refgemm!(C, A, B) = mul!(C, A, B, one(eltype(C)), one(eltype(C)))

# Naive outer-product routine
mygemm!(C, A, B) = mygemm_pji!(C, A, B)
mygemm_name = "mygemm_pji!"

# Big tile which is good when we don't have a vectorized microkernel
mr, nr = 48, 48
# small tile which we will use for vectorized microkernel
# mr, nr = 4, 4
mygemm!(C, A, B) = mygemm_ji_pji!(C, A, B, mr, nr)
mygemm_name = "mygemm_ji_pji!"

# vectorized 4x4 microkernel
mr, nr = 4, 4
mygemm!(C, A, B) = mygemm_ji_microknl_4x4!(C, A, B)
mygemm_name = "mygemm_ji_microknl_4x4!"

# vectorized mr x nr microkernel
mr, nr = 24, 8
vl = 8 # Vector length

# mr must be an integer multiple of the vector length
@assert mod(mr, vl) == 0

mygemm!(C, A, B) = mygemm_ji_microknl!(C, A, B, Val(mr), Val(nr), Val(vl))
mygemm_name = "mygemm_ji_microknl_!"

num_reps = 10

# What precision numbers to use
FloatType = Float32
# FloatType = Float64

@printf("size |      reference      |           %s\n", mygemm_name)
@printf("     |   seconds   GFLOPS  |   seconds   GFLOPS     diff\n")

# Size of square matrix to consider
for nmk in 48:48:480
  n = m = k = nmk
  @printf("%4d |", nmk)

  gflops = 2 * m * n * k * 1e-09

  # Create some random initial data
  A = rand(FloatType, m, k)
  B = rand(FloatType, k, n)
  C = rand(FloatType, m, n)

  # Make a copy of C for resetting data later
  C_old = copy(C)

  # "truth"
  C_ref = A * B + C

  # Compute the reference timings
  best_time = typemax(FloatType)
  for iter = 1:num_reps
    # Reset C to the original data
    C .= C_old;
    run_time = @elapsed refgemm!(C, A, B);
    best_time = min(run_time, best_time)
  end
  # Make sure that we have the right answer!
  @assert C ≈ C_ref
  best_perf = gflops / best_time

  # Print the reference implementation timing
  @printf("  %4.2e %8.2f  |", best_time, best_perf)

  # Compute the timing for mygemm! implementation
  best_time = typemax(FloatType)
  for iter = 1:num_reps
    # Reset C to the original data
    C .= C_old;
    run_time = @elapsed mygemm!(C, A, B);
    best_time = min(run_time, best_time)
  end
  best_perf = gflops / best_time

  # Compute the error (difference between our implementation and the reference)
  diff = norm(C - C_ref, Inf)

  # Print mygemm! implementations
  @printf("  %4.2e %8.2f   %.2e", best_time, best_perf, diff)

  @printf("\n")
end
