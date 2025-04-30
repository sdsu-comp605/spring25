# 34) HW5 Solution

## Assignment steps:

1. (25%) Write your own correct `knl_reversevecs_inplace!` kernel

Solution:

```julia
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

```

Note that in the kernel above we use two temporaries. They are not both necessary, we could have used only one temporary, by creating the `a1` temporary, `a1 = a[i]`, then assigning `a[i] = a[N - i + 1]` directly, and finally assigning `a[N - i + 1] = a1`.

2. (15%) Add testing and timing of this kernel. Your testing should include at least a correctness check.

Solution:

This is done in the code provided in [`julia_codes/hw5_sol/reverse_vec_sol.jl`](https://github.com/sdsu-comp605/spring25/tree/main/julia_codes/hw5_sol/reverse_vec_sol.jl).

Correctness checks are performed using the `@assert` macro, comparing our computed values with the reference value `a_ref`.


```{literalinclude} ../julia_codes/hw5_sol/reverse_vec_sol.jl
:language: julia
:linenos: true
```

3.  (10%) In [`Report.ipynb`](Report.ipynb) explain in your own words what problem the `knl_reversevecs_inplace_bad!` version has.

Solution:
Simply put, the `knl_reversevecs_inplace_bad!` kernel does not use any intermediate temporary variable to swap two entries of an array, which results in incorrect (garbage) values. From the side of GPU computing aspects, it can also lead to race conditions.

4. (10%) In [`Report.ipynb`](Report.ipynb) provide a bandwidth analysis of the correct `knl_reversevecs_inplace!` kernel

Solution:

The bandwidth analysis for the `knl_reversevecs_inplace!` kernel is the same as for the `knl_reversevecs_inplace_bad!` kernel given, that is,


```math
\textrm{bandwidth} = 2 N * {\textrm{sizeof}(T)} / \textrm{time}
```
where ${\tt \textrm{time}}$ is the kernel runtime in seconds, and $\textrm{T}$ is the float type you are executing with (it could be `Float32`, `Float64`, etc.), since we need to load all the data once and write all the data once. We can also divide this by $1024^3$ to obtain Gigabytes per second (Gib/s).

5. (40%) In [`Report.ipynb`](Report.ipynb) add a performance analysis that should produce at least the following two figures and related commentary. Figures that should be included:
  5.1) Bandwidth vs problem size $N$ (you need to test for different values of $N$) for `Float64` and `Float32`.
  5.2) A roofline plot (reference [Lecture 6](https://sdsu-comp605.github.io/spring25/lectures/module2-1_measuring_performance.html)).

  ![HW5 solution: Fig 5.1](../img/bandwidth_F64_Vs_F32.png "HW5 solution: Fig 5.1")

  For the roofline plot, we define:

  ```math
  \textrm{FLOPs} = 4 N
  ```
  Since each assignment that involves floating-point numbers (as opposed to binary integer operations) counts as FLOPs.

  ```math
  \textrm{intensity} =  \frac{\textrm{FLOPs}}{ \textrm{Memory moved}} = \frac{4 N}{ 2 N  {\textrm{sizeof}(T)}}
  ```

  ```math
  \textrm{work} = 4 N  {\textrm{sizeof}(T)} / 1024^3
  ```

  and

  ```math
  \textrm{rate} = \textrm{work} / \textrm{execution time}.
  ```

  Hence, for this particular problem, our roofline plots don't really look like "roofs", rather more like streight walls.

  ![HW5 solution: Fig 5.1](../img/roofline.png "HW5 solution: Fig 5.1")
