using CUDA

# generate some data on the CPU (host array, h_array)
h_array = rand(Float32, 1024)

# allocate on the GPU (device array, d_array)
d_array = CuArray{Float32}(undef, 1024)

# copy from the CPU to the GPU
copyto!(d_array, h_array)

# download/transfer back to the CPU and verify using the Array casting of the device array
@test h_array == Array(d_array)
