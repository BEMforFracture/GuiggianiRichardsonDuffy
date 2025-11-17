import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using ForwardDiff

# Configuration
x̂ = SVector(0.5, 0.5)  # Source point in reference coordinates
n_rho = 10          # Number of quadrature points in rho-direction
n_theta = 15         # Number of quadrature points in theta-direction
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

ref_domain = Inti.reference_domain(el)

# Reference value
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
SK = GRD.SplitKernel(K_base)

s = Inti.singularity_order(SK)
S = s + 1
sorder = Val(S)

ori = 1

method = GRD.AutoDiffExpansion()
acc = GRD.guiggiani_singular_integral(SK, û, x̂, el, ori, quad_rho, quad_theta, method)

@info "Computed integral: $acc"
@info "Expected integral: $expected_I"
@info "Relative error: $(norm(acc - expected_I)/ norm(expected_I))"
