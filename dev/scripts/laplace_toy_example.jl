import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

x̂ = SVector(0.5, 0.5)

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)
p = 1
û = ξ -> Inti.lagrange_basis(typeof(el))(ξ)[p]

K = GRD.SplitLaplaceHypersingular

D = Dict{Symbol, Tuple{Function, Function}}()

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :analytical, name = :LaplaceHypersingular)
D[:analytical] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_1)
D[:semi_analytical_lvl_1] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_2)
D[:semi_analytical_lvl_2] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :full_richardson)
D[:full_richardson] = (F₋₂, F₋₁)

N = 1000
θs = range(0, 2π, length = N)

fig = Figure(; size = (1200, 800))
ax = Axis(fig[1, 1]; xlabel = "θ", ylabel = "Laurent Coefficients", title = "Laplace Hypersingular Kernel Laurent Coefficients")

lw = 4

for (method, (F₋₂, F₋₁)) in D
	lines!(ax, θs, F₋₂.(θs), label = "F₋₂ $method", linewidth = lw)
	lines!(ax, θs, F₋₁.(θs), label = "F₋₁ $method", linestyle = :dash, linewidth = lw)
end
axislegend(ax; position = :rt)
GLMakie.save("./dev/figures/laplace_hypersingular_laurent_coeffs_all_methods.png", fig)
