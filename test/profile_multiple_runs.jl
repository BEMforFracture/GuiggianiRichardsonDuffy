"""
Profile guiggiani_singular_integral multiple times to get stable timing statistics.
"""

using Inti
using Gmsh
using GuiggianiRichardsonDuffy
import GuiggianiRichardsonDuffy as GRD
using LinearAlgebra
using StaticArrays
using Statistics

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

# Create mesh
a = 1.0
b = 0.5
meshsize = 0.1

mesh = create_ellipse(a, b, meshsize)
crack_mesh = view(mesh, Inti.Domain(e -> "C" in Inti.labels(e), mesh))

# Extract first element
el_list = collect(Inti.elements(crack_mesh))
el = first(el_list)

# Simple constant density function
û = xi -> 1

# Get element orientation
ori = 1

# Kernel: Elastostatic hypersingular
K = Inti.HyperSingularKernel(Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0))

# One singular point (interior point in reference element)
x̂ = SVector(0.3, 0.3)  # local reference coords

# Quadrature rules for polar decomposition
quad_rho = Inti.GaussLegendre(10)
quad_theta = Inti.GaussLegendre(20)

println("="^70)
println("PROFILING: Multiple runs to get stable statistics")
println("="^70)
println("Element type: $(typeof(el))")
println("Singularity point: $x̂")
println("Kernel: Elastostatic hypersingular")
println()

# Warmup
println("Running warmup...")
_ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, GRD.FullRichardsonExpansion())
_ = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))

# Multiple runs
N_runs = 1000
times_grd = Float64[]
times_inti = Float64[]

println("\nRunning $N_runs iterations...")
for i in 1:N_runs
    t_grd = @elapsed _ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, GRD.FullRichardsonExpansion())
    t_inti = @elapsed _ = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))
    push!(times_grd, t_grd)
    push!(times_inti, t_inti)
    print(".")
end
println()

println("\n" * "="^70)
println("STATISTICS OVER $N_runs RUNS")
println("="^70)

mean_grd = mean(times_grd)
std_grd = std(times_grd)
min_grd = minimum(times_grd)
max_grd = maximum(times_grd)

mean_inti = mean(times_inti)
std_inti = std(times_inti)
min_inti = minimum(times_inti)
max_inti = maximum(times_inti)

println("\nGRD.guiggiani_singular_integral (FullRichardson):")
println("  Mean: $(round(mean_grd * 1e6; digits=1)) μs")
println("  Std:  $(round(std_grd * 1e6; digits=1)) μs")
println("  Min:  $(round(min_grd * 1e6; digits=1)) μs")
println("  Max:  $(round(max_grd * 1e6; digits=1)) μs")

println("\nInti.guiggiani_singular_integral (Val(-2)):")
println("  Mean: $(round(mean_inti * 1e6; digits=1)) μs")
println("  Std:  $(round(std_inti * 1e6; digits=1)) μs")
println("  Min:  $(round(min_inti * 1e6; digits=1)) μs")
println("  Max:  $(round(max_inti * 1e6; digits=1)) μs")

println("\n" * "="^70)
println("RATIO ANALYSIS")
println("="^70)
ratio_mean = mean_grd / mean_inti
ratio_min = min_grd / max_inti
ratio_max = max_grd / min_inti

println("Mean ratio (GRD/Inti):     $(round(ratio_mean; digits=2))x")
println("Best case ratio (min/max): $(round(ratio_min; digits=2))x")
println("Worst case ratio (max/min): $(round(ratio_max; digits=2))x")

# Identify which runs are fastest for each method
grd_best_idx = argmin(times_grd)
inti_best_idx = argmin(times_inti)
println("\nFastest GRD run: $(round(times_grd[grd_best_idx] * 1e6; digits=1)) μs (run #$grd_best_idx)")
println("Fastest Inti run: $(round(times_inti[inti_best_idx] * 1e6; digits=1)) μs (run #$inti_best_idx)")

# Final verification
println("\n" * "="^70)
W_grd = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, GRD.FullRichardsonExpansion())
W_inti = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))
error_rel = norm(W_grd - W_inti) / (norm(W_inti) + 1e-15)
println("Final relative error: $error_rel")
println("="^70)

### do the same but for auto diff method

t_grd_ad = @timed W_grd_ad = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, GRD.AutoDiffExpansion())
println("\nGRD.guiggiani_singular_integral with AutoDiffExpansion:")
println("Time: $(round(t_grd_ad.time * 1e6; digits=2)) μs")
println("Allocated bytes: $(round(t_grd_ad.bytes / 1024; digits=2)) KB")
error = norm(W_grd_ad - W_inti) / (norm(W_inti) + 1e-15)
println("Relative error vs Inti: $error")
