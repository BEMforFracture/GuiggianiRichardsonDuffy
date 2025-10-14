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

maxeval = 100

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :analytical, name = :LaplaceHypersingular)
D[:analytical] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_1)
D[:semi_analytical_lvl_1] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_2, maxeval = maxeval)
D[:semi_analytical_lvl_2] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :full_richardson, maxeval = maxeval)
D[:full_richardson] = (F₋₂, F₋₁)

N = 1000
θs = range(0, 2π, length = N)

fig = Figure(; size = (1200, 800))
ax = Axis(fig[1, 1]; xlabel = "θ", ylabel = "Laurent Coefficients", title = "Laplace Hypersingular Kernel Laurent Coefficients, Richardson max eval = $maxeval")

for (method, (F₋₂, F₋₁)) in D
	p = 2
	if method != :analytical
		error_F₋₂ = norm(D[:analytical][1].(θs) - F₋₂.(θs), p) / norm(D[:analytical][1].(θs), p)
		@info "Relative error (order $p) F₋₂ ($method vs analytical): $(maximum(error_F₋₂))"
		error_F₋₁ = norm(D[:analytical][2].(θs) - F₋₁.(θs), p) / norm(D[:analytical][2].(θs), p)
		@info "Relative error (order $p) F₋₁ ($method vs analytical): $(maximum(error_F₋₁))"
		lines!(ax, θs, F₋₂.(θs); label = "F₋₂ $method (error = $(round(error_F₋₂, sigdigits=3)))", linewidth = 4)
		lines!(ax, θs, F₋₁.(θs); label = "F₋₁ $method (error = $(round(error_F₋₁, sigdigits=3)))", linewidth = 4, linestyle = :dash)
	end
end

method = :analytical
F₋₂, F₋₁ = D[method]
lines!(ax, θs, F₋₂.(θs); label = "F₋₂ $method", linewidth = 4)
lines!(ax, θs, F₋₁.(θs); label = "F₋₁ $method", linewidth = 4, linestyle = :dash)

axislegend(ax; position = :rt)
GLMakie.save("./dev/figures/laplace_hypersingular_laurent_coeffs_all_methods.png", fig)

fig
