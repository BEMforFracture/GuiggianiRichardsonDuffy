using Inti
using Gmsh
using LinearAlgebra
using StaticArrays
using Profile
using ProfileView
import GuiggianiRichardsonDuffy as GRD

# Create simple mesh
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

println("=" ^ 80)
println("Profiling AutoDiff et SemiRichardson")
println("=" ^ 80)

# Setup
a, b, meshsize = 1.0, 0.5, 0.1
mesh = create_ellipse(a, b, meshsize)
crack_mesh = view(mesh, Inti.Domain(e -> "C" in Inti.labels(e), mesh))
el = first(Inti.elements(crack_mesh))

û = xi -> 1
ori = 1
K = Inti.HyperSingularKernel(Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0))
x̂ = SVector(0.3, 0.3)

quad_rho = Inti.GaussLegendre(10)
quad_theta = Inti.GaussLegendre(20)

method_semi = GRD.SemiRichardsonExpansion()
method_autodiff = GRD.AutoDiffExpansion()

# Warmup
println("\nWarmup...")
for i in 1:10
    GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_semi)
    GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_autodiff)
end

println("\n1. Profiling SemiRichardson (100 iterations)...")
Profile.clear()
@profile for i in 1:100
    GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_semi)
end
println("   Voir ProfileView pour les détails")

println("\n2. Profiling AutoDiff (100 iterations)...")
Profile.clear()
@profile for i in 1:100
    GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_autodiff)
end
println("   Voir ProfileView pour les détails")

println("\n3. Comptage des allocations")
println("\nSemiRichardson:")
@time GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_semi)
allocs_semi = @allocated GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_semi)
println("   Allocations: $(allocs_semi / 1024) KB")

println("\nAutoDiff:")
@time GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_autodiff)
allocs_autodiff = @allocated GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method_autodiff)
println("   Allocations: $(allocs_autodiff / 1024) KB")

println("\n" ^ 80)
println("Pour visualiser le profiling:")
println("  ProfileView.view()")
println("=" ^ 80)
