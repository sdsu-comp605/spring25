using CUDA

a = CuArray(rand(100))
b = CuArray(rand(100))
c = a + b
