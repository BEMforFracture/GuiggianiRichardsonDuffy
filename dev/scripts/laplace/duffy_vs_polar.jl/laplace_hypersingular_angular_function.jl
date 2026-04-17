import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using ForwardDiff

# Configuration
x̂ = SVector(0.3, 0.3)  # Source point in reference coordinates

# Method for Laurent coefficients
method = GRD.AnalyticalExpansion()

# Quadrature parameters
n_rho = 10
quad_rho = Inti.GaussLegendre(n_rho)
target_rel_tol = 1e-8
nmin_quad = 2
nmax_quad = 120

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

ori = 1

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
K = GRD.SplitKernel(K_base)
Kprod = (qx, qy) -> prod(K(qx, qy))

# Polar coordinates setup
K_polar = GRD.polar_kernel_fun(Kprod, el, û, x̂, ori)
ref_domain = Inti.reference_domain(el)
ρ_max_fun = GRD.rho_fun(ref_domain, x̂)

# Compute Laurent coefficients
ℒ = GRD.laurents_coeffs(K_base, el, ori, û, x̂, method)

decompo = Inti.polar_decomposition(ref_domain, x̂)

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

function integrate_gauss_interval(f, a, b, n)
	quad = Inti.GaussLegendre(n)
	mid = (a + b) / 2
	half = (b - a) / 2
	return quad() do (t,)
		half * f(mid + half * t)
	end
end

function min_quad_points_successive(f, a, b; tol = 1e-8, nmin = 2, nmax = 120)
	I_prev = integrate_gauss_interval(f, a, b, nmin)
	last_rel = Inf
	for n in (nmin + 1):nmax
		I_curr = integrate_gauss_interval(f, a, b, n)
		last_rel = abs(I_curr - I_prev) / max(abs(I_curr), eps())
		if last_rel < tol
			return n, last_rel
		end
		I_prev = I_curr
	end
	return nmax, last_rel
end

N = 10000
θs = range(0, stop = 2π, length = N)
fig = Figure(; size = (800, 600))
ax11 = Axis(fig[1, 1]; xlabel = "θ", ylabel = "G₁(θ)")
lines!(ax11, θs, G₁.(θs), label = "G₁(θ)")

θ₁ = decompo[1][1]
θ₂ = decompo[2][1]
θ₃ = decompo[3][1]
θ₄ = decompo[4][1]

θ_bounds = [θ₁, θ₂, θ₃, θ₄, θ₁ + 2π]
nθ_G1 = Int[]
nθ_G2 = Int[]
for T in 1:4
	θa = θ_bounds[T]
	θb = θ_bounds[T + 1]
	n1, rel1 = min_quad_points_successive(
		θ -> G₁(mod2pi(θ)),
		θa,
		θb;
		tol = target_rel_tol,
		nmin = nmin_quad,
		nmax = nmax_quad,
	)
	n2, rel2 = min_quad_points_successive(
		θ -> G₂(mod2pi(θ)),
		θa,
		θb;
		tol = target_rel_tol,
		nmin = nmin_quad,
		nmax = nmax_quad,
	)
	push!(nθ_G1, n1)
	push!(nθ_G2, n2)
	println("Secteur T=$T | tol=$target_rel_tol | G1: n_theta=$n1 (rel=$rel1) | G2: n_theta=$n2 (rel=$rel2)")
end

G1_vals = G₁.(θs)
G2_vals = G₂.(θs)
G1_ymin, G1_ymax = extrema(G1_vals)
G2_ymin, G2_ymax = extrema(G2_vals)
G1_ylabel = G1_ymax - 0.08 * (G1_ymax - G1_ymin)
G2_ylabel = G2_ymax - 0.08 * (G2_ymax - G2_ymin)

ax11.title = "G₁(θ) | tol=$(target_rel_tol)"

vlines!(ax11, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

for T in 1:4
	θa = θ_bounds[T]
	θb = θ_bounds[T + 1]
	θlabel = θa + 0.04 * (θb - θa)
	text!(ax11, θlabel, G1_ylabel; text = "T=$T: nθ=$(nθ_G1[T])", align = (:left, :top), fontsize = 12, color = :black)
end

ax21 = Axis(fig[2, 1]; xlabel = "θ", ylabel = "G₂(θ)")
lines!(ax21, θs, G2_vals, label = "G₂(θ)")
vlines!(ax21, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

for T in 1:4
	θa = θ_bounds[T]
	θb = θ_bounds[T + 1]
	θlabel = θa + 0.04 * (θb - θa)
	text!(ax21, θlabel, G2_ylabel; text = "T=$T: nθ=$(nθ_G2[T])", align = (:left, :top), fontsize = 12, color = :black)
end

ax31 = Axis(fig[3, 1]; xlabel = "θ", ylabel = "ρ(θ)")
lines!(ax31, θs, ρ_max_fun.(θs), label = "ρ(θ)")
vlines!(ax31, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax12 = Axis(fig[1, 2]; xlabel = "θ", ylabel = "G₁'(θ)")
lines!(ax12, θs, dG₁.(θs), label = "G₁'(θ)")
vlines!(ax12, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax22 = Axis(fig[2, 2]; xlabel = "θ", ylabel = "G₂'(θ)")
lines!(ax22, θs, dG₂.(θs), label = "G₂'(θ)")
vlines!(ax22, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

ax32 = Axis(fig[3, 2]; xlabel = "θ", ylabel = "ρ'(θ)")
lines!(ax32, θs, dρ_max.(θs), label = "ρ'(θ)")
vlines!(ax32, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

window = display(GLMakie.Screen(), fig)

# GLMakie.save("./dev/figures/laplace/laplace_hypersingular_angular_function.png", fig)
