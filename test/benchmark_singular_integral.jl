"""
Benchmark: guiggiani_singular_integral performance (single element)
Compare les 4 méthodes vs Inti sur un élément unique
"""

using Inti
using Gmsh
using LinearAlgebra
using StaticArrays
using Statistics

import GuiggianiRichardsonDuffy as GRD

GC.gc()

# Create ellipse mesh
function create_ellipse(a, b, meshsize)
    gmsh.initialize()
    gmsh.model.add("ellipse")
    lc = meshsize
    el = gmsh.model.occ.addEllipse(0.0, 0.0, 0.0, a, b, -1)
    curve_loop = gmsh.model.occ.addCurveLoop([el])
    crack = gmsh.model.occ.addPlaneSurface([el], -1)
    gmsh.model.occ.synchronize()
    pg_crack = gmsh.model.addPhysicalGroup(2, [crack], -1, "C")
    pg_front = gmsh.model.addPhysicalGroup(1, [el], -1, "F")
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)
    mesh = Inti.import_mesh(; dim = 3)
    gmsh.finalize()
    return mesh
end

println("="^70)
println("BENCHMARK: guiggiani_singular_integral (single element)")
println("="^70)

# Setup
a = 1.0
b = 0.5
meshsize = 0.1

mesh = create_ellipse(a, b, meshsize)
crack_mesh = view(mesh, Inti.Domain(e -> "C" in Inti.labels(e), mesh))

el_list = collect(Inti.elements(crack_mesh))
el = first(el_list)

û = xi -> 1
ori = 1
K = Inti.HyperSingularKernel(Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0))
x̂ = SVector(0.3, 0.3)

quad_rho = Inti.GaussLegendre(10)
quad_theta = Inti.GaussLegendre(20)

println("Element type: $(typeof(el))")
println("Singularity point: $x̂")
println("Kernel: Elastostatic hypersingular")
println()

# Test all 4 methods
method_full = GRD.FullRichardsonExpansion()
method_semi = GRD.SemiRichardsonExpansion()
method_autodiff = GRD.AutoDiffExpansion()
method_analytical = GRD.AnalyticalExpansion()

# Warmup
println("Running warmup...")
_ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_full)
_ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_semi)
_ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_autodiff)
_ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_analytical)
_ = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))
println("Warmup complete.\n")

# Benchmark
N_runs = 10
times_full = Float64[]
times_semi = Float64[]
times_autodiff = Float64[]
times_analytical = Float64[]
times_inti = Float64[]

println("Running $N_runs iterations...")
for i in 1:N_runs
    GC.gc()
    t_full = @elapsed _ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_full)
    GC.gc()
    t_semi = @elapsed _ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_semi)
    GC.gc()
    t_autodiff = @elapsed _ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_autodiff)
    GC.gc()
    t_analytical = @elapsed _ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_analytical)
    GC.gc()
    t_inti = @elapsed _ = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))
    GC.gc()
    push!(times_full, t_full)
    push!(times_semi, t_semi)
    push!(times_autodiff, t_autodiff)
    push!(times_analytical, t_analytical)
    push!(times_inti, t_inti)
    
    if i % 10 == 0
        print(".")
    end
end
println()

# Statistics
println("\n" * "="^70)
println("STATISTICS OVER $N_runs RUNS")
println("="^70)

function print_stats(name, times)
    println("\n$name:")
    println("  Mean: $(round(mean(times) * 1e6; digits=1)) μs")
    println("  Std:  $(round(std(times) * 1e6; digits=1)) μs")
    println("  Min:  $(round(minimum(times) * 1e6; digits=1)) μs")
    println("  Max:  $(round(maximum(times) * 1e6; digits=1)) μs")
end

print_stats("FullRichardson", times_full)
print_stats("SemiRichardson", times_semi)
print_stats("AutoDiff", times_autodiff)
print_stats("Analytical", times_analytical)
print_stats("Inti (reference)", times_inti)

# Ratio analysis
println("\n" * "="^70)
println("SPEEDUP vs INTI")
println("="^70)

mean_inti = mean(times_inti)

for (name, times) in [
    ("FullRichardson", times_full),
    ("SemiRichardson", times_semi),
    ("AutoDiff", times_autodiff),
    ("Analytical", times_analytical)
]
    ratio = mean(times) / mean_inti
    println("  $name: $(round(ratio; digits=2))x")
end

# Verification
println("\n" * "="^70)
println("NUMERICAL VERIFICATION")
println("="^70)

W_inti = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))

println("\nRelative errors vs Inti:")
for (name, method) in [
    ("FullRichardson", method_full),
    ("SemiRichardson", method_semi),
    ("AutoDiff", method_autodiff),
    ("Analytical", method_analytical)
]
    W = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method)
    error_rel = norm(W - W_inti) / (norm(W_inti) + 1e-15)
    GC.gc()
    println("  $name: $(error_rel)")
end

println("\n" * "="^70)
