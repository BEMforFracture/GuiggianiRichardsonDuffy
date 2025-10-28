import GuiggianiRichardsonDuffy as GRD
using Inti
using StaticArrays
using Profile
using ProfileView

# Configuration identique au benchmark
x̂ = SVector(0.5, 0.5)

rich_params = GRD.RichardsonParams(
    first_contract = 1e-2,
    contract = 0.5,
    breaktol = Inf,
    atol = 0.0,
    rtol = 0.0,
    maxeval = 5,
)

n_rho = 10
n_theta = 40
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

û = ξ -> 1.0

K_base = Inti.HyperSingularKernel(Inti.Laplace(dim=3))

# Test avec FullRichardson (le plus lent)
method = GRD.FullRichardsonExpansion(rich_params)

println("=== Warming up ===")
# Warm-up
for i in 1:5
    GRD.guiggiani_singular_integral(K_base, û, x̂, el, quad_rho, quad_theta, method)
end

println("\n=== Profiling ===")
# Profile
Profile.clear()
@profile for i in 1:100
    GRD.guiggiani_singular_integral(K_base, û, x̂, el, quad_rho, quad_theta, method)
end

println("\n=== Results ===")
Profile.print(format=:flat, sortedby=:count, mincount=10)

println("\n=== Opening ProfileView (close window to continue) ===")
ProfileView.view()
