import GuiggianiRichardsonDuffy as GRD

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

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)

p = 1
û = ξ -> Inti.lagrange_basis(typeof(el))(ξ)[p]

F₋₂ = θ -> GRD.LaplaceHypersingularClosedFormF₋₂(θ, η, el, û)
F₋₁ = θ -> GRD.LaplaceHypersingularClosedFormF₋₁(θ, η, el, û)

θ = π

println("F₋₂($θ) = ", F₋₂(θ))
println("F₋₁($θ) = ", F₋₁(θ))
