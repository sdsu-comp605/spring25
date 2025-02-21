# What modules / packages do we depend on
using Random
using LinearAlgebra
using Printf
using UnicodePlots

# To ensure repeatability
Random.seed!(777)

# Don't let BLAS use lots of threads (since we are not multi-threaded yet!)
BLAS.set_num_threads(64)

include("mygemm.jl")
include("mygemm_packed.jl")
include("mygemm_packed_MT5.jl")
include("mygemm_packed_MT3.jl")
include("mygemm_packed_MT2.jl")
include("mygemm_packed_MT1.jl")

# C := α * A * B + β * C
refgemm!(C, A, B) = mul!(C, A, B, one(eltype(C)), one(eltype(C)))

# FloatType = Float32
FloatType = Float64

# Block matrix-matrix multiply (microknl)
if FloatType == Float64
    _mc = 96
    _nc = 3024
    _kc = 256
    _mr = 12
    _nr = 4
    _vl = 4
elseif FloatType == Float32
    _mc = 2 * 96
    _nc = 2 * 2016
    _kc = 2 * 256
    _mr = 24
    _nr = 8
    _vl = 8
end
sizes = Sizes{_mc, _nc, _kc, _mr, _nr, _vl}()

# mygemm!(C, A, B) = mygemm!(C, A, B, sizes); mygemm_name = "mygemm!"
# mygemm!(C, A, B) = mygemm_packed!(C, A, B, sizes); mygemm_name = "mygemm_packed!"
# mygemm!(C, A, B) = mygemm_packed_MT1!(C, A, B, sizes); mygemm_name = "mygemm_packed_MT1!"
# mygemm!(C, A, B) = mygemm_packed_MT2!(C, A, B, sizes); mygemm_name = "mygemm_packed_MT2!"
# mygemm!(C, A, B) = mygemm_packed_MT3!(C, A, B, sizes); mygemm_name = "mygemm_packed_MT3!"
mygemm!(C, A, B) = mygemm_packed_MT5!(C, A, B, sizes); mygemm_name = "mygemm_packed_MT5!"

num_reps = 10

@printf("size |      reference      |           %s\n", mygemm_name)
@printf("     |   seconds   GFLOPS  |   seconds   GFLOPS     diff\n")

# Size of square matrix to consider
nmks = 52*48:-48:48
my_data = zeros(length(nmks))
ref_data = zeros(length(nmks))
for (iter, nmk) in enumerate(nmks)
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
  ref_data[iter] = best_perf

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
  my_data[iter] = best_perf

  # Compute the error (difference between our implementation and the reference)
  diff = norm(C - C_ref, Inf)

  # Print mygemm! implementations
  @printf("  %4.2e %8.2f   %.2e", best_time, best_perf, diff)

  @printf("\n")
end
plt = lineplot(nmks, ref_data, name="ref data")
lineplot!(plt, nmks, my_data, name="my data")
display(plt)
