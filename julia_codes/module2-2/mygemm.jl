# matrix times row vector (update rows of `C`) with inner dot product
function mygemm_ijp!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for i = 1:m
    for j = 1:n
      for p = 1:k
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# matrix times row vector (update rows of `C`) with inner axpy
function mygemm_ipj!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for i = 1:m
    for p = 1:k
      for j = 1:n
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# Rank one update (repeatedly update all elements of `C`) with outer product
# using axpy with rows of `B`
function mygemm_pij!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for p = 1:k
    for i = 1:m
      for j = 1:n
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# Rank one update (repeatedly update all elements of `C`) with outer product
# using axpy with columns of `A`
function mygemm_pji!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for p = 1:k
    for j = 1:n
      for i = 1:m
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# matrix times column vector (update columns of `C`) with inner axpy
function mygemm_jpi!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for j = 1:n
    for p = 1:k
      for i = 1:m
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end

# matrix times column vector (update columns of `C`) with inner dot product
function mygemm_jip!(C, A, B)
  m, k = size(A)
  _, n = size(B)
  @assert size(B, 1) == k
  @assert size(C) == (m, n)

  for j = 1:n
    for i = 1:m
      for p = 1:k
        @inbounds C[i, j] += A[i, p] * B[p, j]
      end
    end
  end
end
