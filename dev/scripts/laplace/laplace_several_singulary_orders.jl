import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

# INPUTS

x̂ = SVector(0.1, 0.1) # source point in reference coordinates

### Richardson extrapolation parameters
maxeval = 10
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

maxeval_in_loop = 100

# END INPUTS

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)
ref_domain = Inti.reference_domain(el)

û = ξ -> 1.0

K³ = GRD.SplitLaplaceHypersingular
K² = GRD.SplitLaplaceAdjointDoubleLayer
K¹ = GRD.SplitLaplaceSingleLayer

ℒ³ = GRD.laurents_coeffs(K³, el, û, x̂; expansion = :auto_diff, sorder = Val(-3))
ℒ² = GRD.laurents_coeffs(K², el, û, x̂; expansion = :auto_diff, sorder = Val(-2))
ℒ¹ = GRD.laurents_coeffs(K¹, el, û, x̂; expansion = :auto_diff, sorder = Val(-1))
