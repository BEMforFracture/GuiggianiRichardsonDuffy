"""
MWE: Breakdown du coût par étape pour chaque méthode

Pour une itération theta donnée, chaque méthode fait :

FullRichardson:
  1. Richardson sur F(ρ) avec kernel DIRECT       → N_rich évals de K
  2. Quadrature rho avec K_polar DIRECT            → N_rho évals de K

AutoDiff:
  1. ℱ(0) : 1 éval SplitKernel                    → 1 éval SK
  2. ForwardDiff.derivative(ℱ, 0) avec Dual        → ~2 évals SK (dual numbers)
  3. Quadrature rho avec K_polar via polar_kernel_fun → N_rho évals de Kprod=prod∘SK

SemiRichardson:
  1. Partie dominante : A(θ), Â, SK(qx,qx,Â)      → 1 éval SK + algèbre
  2. Richardson sur K_polar via polar_kernel_fun    → N_rich évals de Kprod
  3. Quadrature rho avec K_polar via polar_kernel_fun → N_rho évals de Kprod

Ce script mesure chaque étape individuellement.
"""

using Inti
using Gmsh
using LinearAlgebra
using StaticArrays
using Statistics
using ForwardDiff

import GuiggianiRichardsonDuffy as GRD

# ============================================================================
# SETUP
# ============================================================================

function create_ellipse(a, b, meshsize)
    gmsh.initialize()
    gmsh.model.add("ellipse")
    el = gmsh.model.occ.addEllipse(0.0, 0.0, 0.0, a, b, -1)
    gmsh.model.occ.addCurveLoop([el])
    crack = gmsh.model.occ.addPlaneSurface([el], -1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [crack], -1, "C")
    gmsh.model.addPhysicalGroup(1, [el], -1, "F")
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)
    mesh = Inti.import_mesh(; dim = 3)
    gmsh.finalize()
    return mesh
end

println("="^70)
println("MWE: Breakdown du coût par étape pour chaque méthode")
println("="^70)

mesh = create_ellipse(1.0, 0.5, 0.1)
crack_mesh = view(mesh, Inti.Domain(e -> "C" in Inti.labels(e), mesh))
el = first(collect(Inti.elements(crack_mesh)))

û = xi -> 1
ori = 1
K = Inti.HyperSingularKernel(Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0))
x̂ = SVector(0.3, 0.3)
quad_rho = Inti.GaussLegendre(10)

# Pré-calculs communs
ref_shape = Inti.reference_domain(el)
decompo = Inti.polar_decomposition(ref_shape, x̂)
# Choisir un θ dans le premier secteur angulaire
θ_test = (decompo[1][1] + decompo[1][2]) / 2
ρ_max = decompo[1][3](θ_test)

# ============================================================================
# ÉTAPE 1 : Évaluation du kernel (direct vs SplitKernel vs Kprod)
# ============================================================================
println("\n" * "="^70)
println("ÉTAPE 1 : Évaluation unitaire du kernel")
println("="^70)

x_phys = el(x̂)
jac_x = Inti.jacobian(el, x̂)
nx = Inti._normal(jac_x, ori)
qx = (coords = x_phys, normal = nx)

ŷ = x̂ + 0.1 * SVector(cos(θ_test), sin(θ_test))
y_phys = el(ŷ)
jac_y = Inti.jacobian(el, ŷ)
ny = Inti._normal(jac_y, ori)
qy = (coords = y_phys, normal = ny)

SK = GRD.SplitKernel(K)
Kprod = (qx, qy) -> prod(SK(qx, qy))

N = 10_000

# Direct
t = @elapsed for _ in 1:N; K(qx, qy); end
println("\n  K(qx, qy) direct      : $(round(t/N * 1e9, digits=1)) ns/call")

# SplitKernel
t = @elapsed for _ in 1:N; SK(qx, qy); end
println("  SK(qx, qy)            : $(round(t/N * 1e9, digits=1)) ns/call")

# prod(SK(...))
t = @elapsed for _ in 1:N; prod(SK(qx, qy)); end
println("  prod(SK(qx, qy))      : $(round(t/N * 1e9, digits=1)) ns/call")

# Kprod closure
t = @elapsed for _ in 1:N; Kprod(qx, qy); end
println("  Kprod closure          : $(round(t/N * 1e9, digits=1)) ns/call")

# ============================================================================
# ÉTAPE 2 : K_polar (inline direct vs polar_kernel_fun)
# ============================================================================
println("\n" * "="^70)
println("ÉTAPE 2 : K_polar (inline vs polar_kernel_fun)")
println("="^70)

# Version inline (comme FullRichardson optimisé)
K_polar_inline = function (ρ, θ)
    s_theta, c_theta = sincos(θ)
    ŷ = x̂ + ρ * SVector(c_theta, s_theta)
    y = el(ŷ)
    jac_y = Inti.jacobian(el, ŷ)
    ny = Inti._normal(jac_y, ori)
    qy = (coords = y, normal = ny)
    μ = Inti._integration_measure(jac_y)
    M = K(qx, qy)
    v = û(ŷ)
    return ρ * map(v -> M * v, v) * μ
end

# Version via polar_kernel_fun avec Kprod (comme AutoDiff/SemiRichardson)
K_polar_pkf = GRD.polar_kernel_fun(Kprod, el, û, x̂, ori)

# Version via polar_kernel_fun avec K direct
K_polar_pkf_direct = GRD.polar_kernel_fun(K, el, û, x̂, ori)

ρ_test = 0.05

t = @elapsed for _ in 1:N; K_polar_inline(ρ_test, θ_test); end
println("\n  K_polar inline (direct K)              : $(round(t/N * 1e9, digits=1)) ns/call")

t = @elapsed for _ in 1:N; K_polar_pkf_direct(ρ_test, θ_test); end
println("  polar_kernel_fun(K, ...)               : $(round(t/N * 1e9, digits=1)) ns/call")

t = @elapsed for _ in 1:N; K_polar_pkf(ρ_test, θ_test); end
println("  polar_kernel_fun(Kprod, ...)            : $(round(t/N * 1e9, digits=1)) ns/call")

# ============================================================================
# ÉTAPE 3 : Calcul des coefficients de Laurent
# ============================================================================
println("\n" * "="^70)
println("ÉTAPE 3 : Calcul des coefficients de Laurent (par θ)")
println("="^70)

# FullRichardson
ℒ_full = GRD.laurents_coeffs(K, el, ori, û, x̂, GRD.FullRichardsonExpansion())
t = @elapsed for _ in 1:N; ℒ_full(θ_test); end
println("\n  FullRichardson ℒ(θ)    : $(round(t/N * 1e6, digits=2)) μs/call")

# AutoDiff  
ℒ_ad = GRD.laurents_coeffs(K, el, ori, û, x̂, GRD.AutoDiffExpansion())
t = @elapsed for _ in 1:N; ℒ_ad(θ_test); end
println("  AutoDiff ℒ(θ)          : $(round(t/N * 1e6, digits=2)) μs/call")

# SemiRichardson
ℒ_semi = GRD.laurents_coeffs(K, el, ori, û, x̂, GRD.SemiRichardsonExpansion())
t = @elapsed for _ in 1:N; ℒ_semi(θ_test); end
println("  SemiRichardson ℒ(θ)    : $(round(t/N * 1e6, digits=2)) μs/call")

# Analytical
ℒ_anal = GRD.laurents_coeffs(K, el, ori, û, x̂, GRD.AnalyticalExpansion())
t = @elapsed for _ in 1:N; ℒ_anal(θ_test); end
println("  Analytical ℒ(θ)        : $(round(t/N * 1e6, digits=2)) μs/call")

# ============================================================================
# ÉTAPE 4 : Quadrature radiale (pour un θ fixe)
# ============================================================================
println("\n" * "="^70)
println("ÉTAPE 4 : Quadrature radiale (pour un θ fixe)")
println("="^70)

f₋₂_full, f₋₁_full = ℒ_full(θ_test)
f₋₂_ad, f₋₁_ad = ℒ_ad(θ_test)

# Avec K_polar inline (FullRichardson/Analytical)
t = @elapsed for _ in 1:N
    quad_rho() do (rho_ref,)
        ρ = ρ_max * rho_ref
        ρ < cbrt(eps()) && (return zero(f₋₂_full))
        K_polar_inline(ρ, θ_test) - f₋₂_full / ρ^2 - f₋₁_full / ρ
    end
end
println("\n  Quad rho (K_polar inline)       : $(round(t/N * 1e6, digits=2)) μs")

# Avec K_polar via polar_kernel_fun + Kprod (AutoDiff/SemiRichardson)
t = @elapsed for _ in 1:N
    quad_rho() do (rho_ref,)
        ρ = ρ_max * rho_ref
        ρ < cbrt(eps()) && (return zero(f₋₂_ad))
        K_polar_pkf(ρ, θ_test) - f₋₂_ad / ρ^2 - f₋₁_ad / ρ
    end
end
println("  Quad rho (polar_kernel_fun+Kprod): $(round(t/N * 1e6, digits=2)) μs")

# ============================================================================
# ÉTAPE 5 : Intégrale complète
# ============================================================================
println("\n" * "="^70)
println("ÉTAPE 5 : Intégrale singulière complète (1000 runs)")
println("="^70)

N_runs = 1000

methods = [
    ("FullRichardson", GRD.FullRichardsonExpansion()),
    ("AutoDiff",       GRD.AutoDiffExpansion()),
    ("SemiRichardson", GRD.SemiRichardsonExpansion()),
    ("Analytical",     GRD.AnalyticalExpansion()),
]

quad_theta = Inti.GaussLegendre(20)

# Warmup
for (_, m) in methods
    _ = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, m)
end
_ = Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))

# Benchmark
for (name, method) in methods
    times = [(@elapsed GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, method)) for _ in 1:N_runs]
    println("\n  $name:")
    println("    Mean: $(round(mean(times) * 1e6, digits=1)) μs")
    println("    Min:  $(round(minimum(times) * 1e6, digits=1)) μs")
end

times_inti = [(@elapsed Inti.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_theta, Val(-2))) for _ in 1:N_runs]
println("\n  Inti (reference):")
println("    Mean: $(round(mean(times_inti) * 1e6, digits=1)) μs")
println("    Min:  $(round(minimum(times_inti) * 1e6, digits=1)) μs")

println("\n" * "="^70)
println("CONCLUSION")
println("="^70)
println("""
L'analyse par étape permet d'identifier :
- Si le bottleneck est dans le calcul des coefficients de Laurent (ℒ(θ))
- Ou dans la quadrature radiale (K_polar évalué N_rho fois)
- Ou dans l'évaluation unitaire du kernel (K vs SK vs Kprod)
- Ou dans l'utilisation de polar_kernel_fun vs inline
""")
println("="^70)
