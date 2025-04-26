using Plots
default(linewidth=4, legendfontsize=12)
using CSV, DataFrames

test_results = DataFrame(exec_num = Int[], naive_time = Float64[], mst_time = Float64[], iter_time = Float64[])

open("jred_comp.out", "r")  do f

    output_file = read(f, String)

    # Regular expressions
    # First, split the read file into blocks per Execution
    blocks = split(output_file, "=====================================")

    # Prepare an empty DataFrame
    df = DataFrame(execution = Int[], naive_time = Float64[], recursive_mst_time = Float64[], iterative_time = Float64[])

    # Loop through each block
    for block in blocks
        if occursin(r"Execution \d+ MPI rank", block)
            # Extract Execution number
            exec_match = match(r"Execution (\d+)", block)
            exec_num = parse(Int, exec_match.captures[1])

            # Extract times
            naive_match = match(r"Elapsed time for the naive algorithm:.*?=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+|))"s, block)
            recursive_mst_match = match(r"Elapsed time for the recursive mst algorithm:.*?=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+|))"s, block)
            iterative_match = match(r"Elapsed time for the iterative \(non-recursive\) mst algorithm:.*?=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+|))"s, block)

            if naive_match !== nothing && recursive_mst_match !== nothing && iterative_match !== nothing

                # Populate if matches found

                naive_time = parse(Float64, naive_match.captures[1])
                mst_time = parse(Float64, recursive_mst_match.captures[1])
                iter_time = parse(Float64, iterative_match.captures[1])

                push!(test_results, (exec_num = exec_num, naive_time = naive_time, mst_time = mst_time, iter_time = iter_time))
            end
        end
    end
end

CSV.write("test_results.csv", test_results)

pl = plot(
    test_results.exec_num, test_results.naive_time,
    yscale = :log10,
    xlims = (0,16),
    xlabel = "MPI size",
    ylabel = "Execution time",
    label = "Naive reduce",
    title = "Comparison of different parallel MPI reduce(to-one) algorithms",
    marker=:o
)

plot!(test_results.exec_num, test_results.mst_time, label = "MST (recursive) reduce", marker=:diamond)
plot!(test_results.exec_num, test_results.iter_time, label = "Iterative reduce", marker=:star5)

savefig(pl, "comparison_naive_mst_iter_reduce.png")
