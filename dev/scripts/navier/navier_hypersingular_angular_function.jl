import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using ForwardDiff

# INPUTS

x̂ = SVector(0.01, 0.01) # source point in reference coordinates

# Material properties
μ = 1.0
λ = 1.0

# Richardson parameters
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 8,
)

# Quadrature parameters
n_rho = 10

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
x = el(x̂)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

# Kernel setup
op = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)
K_base = Inti.HyperSingularKernel(op)
K = GRD.SplitKernel(K_base)

K_polar = GRD.polar_kernel_fun(K_base, el, û, x̂)

quad_rho = Inti.GaussLegendre(n_rho)

ref_domain = Inti.reference_domain(el)
ρ_max_fun = GRD.rho_fun(ref_domain, x̂)

decompo = Inti.polar_decomposition(ref_domain, x̂)

ℒ = GRD.laurents_coeffs(K, el, û, x̂, GRD.AutoDiffExpansion())

function G₁(θ)
	f₋₂, f₋₁ = ℒ(θ)
	ρ_max = ρ_max_fun(θ)
	I_rho = quad_rho() do (rho_ref,)
		ρ = ρ_max * rho_ref
		return K_polar(ρ, θ) - f₋₂ / ρ^2 - f₋₁ / ρ
	end
	return I_rho * ρ_max
end

function G₂(θ)
	f₋₂, f₋₁ = ℒ(θ)
	ρ_max = ρ_max_fun(θ)
	return f₋₁ * log(ρ_max) - f₋₂ / ρ_max
end

function G(θ)
	return G₁(θ) + G₂(θ)
end

function dG₁(θ)
	return ForwardDiff.derivative(G₁, θ)
end

function dG₂(θ)
	return ForwardDiff.derivative(G₂, θ)
end

function dG(θ)
	return dG₁(θ) + dG₂(θ)
end

function dρ_max(θ)
	return ForwardDiff.derivative(ρ_max_fun, θ)
end

N = 10000
θs = range(0, stop = 2π, length = N)
fig = Figure(; size = (800, 600))
ax11 = Axis(fig[1, 1]; xlabel = "θ", ylabel = "G₁(θ)")

colors = [[:blue, :orange, :green], [:red, :purple, :brown], [:pink, :gray, :cyan]]

for i in 1:3
	for j in 1:3
		g = [G₁(θ)[i, j] for θ in θs]
		lines!(ax11, θs, g, label = "G₁(θ)[$i,$j]", color = colors[i][j])
	end
end
θ₁ = decompo[1][1]
θ₂ = decompo[2][1]
θ₃ = decompo[3][1]
θ₄ = decompo[4][1]

vlines!(ax11, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax21 = Axis(fig[2, 1]; xlabel = "θ", ylabel = "G₂(θ)")
for i in 1:3
	for j in 1:3
		g = [G₂(θ)[i, j] for θ in θs]
		lines!(ax21, θs, g, label = "G₂(θ)[$i,$j]", color = colors[i][j])
	end
end

vlines!(ax21, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax31 = Axis(fig[3, 1]; xlabel = "θ", ylabel = "ρ(θ)")
lines!(ax31, θs, ρ_max_fun.(θs), label = "ρ(θ)")
vlines!(ax31, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax12 = Axis(fig[1, 2]; xlabel = "θ", ylabel = "G₁'(θ)")
for i in 1:3
	for j in 1:3
		g = [dG₁(θ)[i, j] for θ in θs]
		lines!(ax12, θs, g, label = "G₁'(θ)[$i,$j]", color = colors[i][j])
	end
end
vlines!(ax12, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax22 = Axis(fig[2, 2]; xlabel = "θ", ylabel = "G₂'(θ)")
for i in 1:3
	for j in 1:3
		g = [dG₂(θ)[i, j] for θ in θs]
		lines!(ax22, θs, g, label = "G₂'(θ)[$i,$j]", color = colors[i][j])
	end
end
vlines!(ax22, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax32 = Axis(fig[3, 2]; xlabel = "θ", ylabel = "ρ'(θ)")
lines!(ax32, θs, dρ_max.(θs), label = "ρ'(θ)")
vlines!(ax32, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

display(fig)

GLMakie.save("./dev/figures/navier/navier_hypersingular_angular_function.png", fig)
