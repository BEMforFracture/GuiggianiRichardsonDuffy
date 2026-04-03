"""
Benchmark: adaptive_correction performance comparison
Compares the 4 methods (FullRichardson, SemiRichardson, AutoDiff, Analytical) vs Inti
"""

using Inti
using LinearAlgebra
using Statistics
using SparseArrays
using Gmsh

import GuiggianiRichardsonDuffy as GRD

println("="^70)
println("BENCHMARK: adaptive_correction performance")
println("="^70)

# Load mesh
testdir = @__DIR__
projectdir = dirname(testdir)
mesh_path = joinpath(projectdir, "assets", "meshes_template", "disks", "disk_infinite_media.msh")
msh = GRD.@suppress_output Inti.import_mesh(mesh_path; dim = 3)
Γ_msh = view(msh, Inti.Domain(e -> "C" in Inti.labels(e), msh))

# Setup quadrature
Q = Inti.Quadrature(Γ_msh; qorder = 2)

# Richardson parameters
rich_params = GRD.RichardsonParams(maxeval = 10)

# Methods to test
methods = [
    ("FullRichardson", GRD.FullRichardsonExpansion(rich_params)),
    ("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params)),
    ("AutoDiff", GRD.AutoDiffExpansion()),
]

# Test kernels
ops = [
    ("Laplace", Inti.Laplace(dim = 3)),
    ("Elastostatic", Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)),
]

println("\nSetup:")
println("  Mesh: $(length(Q.qnodes)) quadrature nodes")
println("  Richardson maxeval: 10")
println()

# Warmup
println("Warmup...")
for (op_name, op) in ops
    K = Inti.HyperSingularKernel(op)
    iop = Inti.IntegralOperator(K, Q)
    
    for (method_name, method) in methods
        _ = GRD.adaptive_correction(iop; method = method)
    end
    
    _ = Inti.adaptive_correction(iop)
end
println("Warmup complete.\n")

# Benchmark
N_runs = 1

for (op_name, op) in ops
    println("="^70)
    println("OPERATOR: $op_name")
    println("="^70)
    
    K = Inti.HyperSingularKernel(op)
    iop = Inti.IntegralOperator(K, Q)
    
    # Store times
    times_grd = Dict{String, Vector{Float64}}()
    for (method_name, _) in methods
        times_grd[method_name] = Float64[]
    end
    times_inti = Float64[]
    
    println("\nRunning $N_runs iterations...")
    for i in 1:N_runs
        # GRD methods
        for (method_name, method) in methods
            t = @elapsed _ = GRD.adaptive_correction(iop; method = method)
            push!(times_grd[method_name], t)
        end
        
        # Inti reference
        t_inti = @elapsed _ = Inti.adaptive_correction(iop)
        push!(times_inti, t_inti)
        
        if i % 100 == 0
            print(".")
        end
    end
    println()
    
    # Statistics
    println("\n" * "-"^70)
    println("STATISTICS OVER $N_runs RUNS")
    println("-"^70)
    
    function print_stats(name, times)
        println("\n$name:")
        println("  Mean: $(round(mean(times) * 1e3; digits=2)) ms")
        println("  Std:  $(round(std(times) * 1e3; digits=2)) ms")
        println("  Min:  $(round(minimum(times) * 1e3; digits=2)) ms")
        println("  Max:  $(round(maximum(times) * 1e3; digits=2)) ms")
    end
    
    for (method_name, _) in methods
        print_stats(method_name, times_grd[method_name])
    end
    
    print_stats("Inti (reference)", times_inti)
    
    # Ratio analysis
    println("\n" * "-"^70)
    println("SPEEDUP vs INTI")
    println("-"^70)
    
    mean_inti = mean(times_inti)
    
    for (method_name, _) in methods
        ratio = mean(times_grd[method_name]) / mean_inti
        println("  $method_name: $(round(ratio; digits=2))x")
    end
    
    println()
end

println("="^70)
println("BENCHMARK COMPLETE")
println("="^70)
