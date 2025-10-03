using GuiggianiRichardsonDuffy
using Inti
using StaticArrays

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

η = SVector(0.5, 0.5)

τ = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(τ)

p = 1
û = ξ -> Inti.lagrange_basis(typeof(τ))(ξ)[p]

ν = 0.3 # Poisson ratio
i, j, k = 1, 1, 1 # indices of the derivative of the kernel

F₋₂ = θ -> navier_hypersingular_F₋₂(θ, τ, i, j, k, û, η, ν)
F₋₁ = θ -> navier_hypersingular_F₋₁(θ, τ, i, j, k, û, η, ν)

θ = π

println("F₋₂($θ) = ", F₋₂(θ))
println("F₋₁($θ) = ", F₋₁(θ))
