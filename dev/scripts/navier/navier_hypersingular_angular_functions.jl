import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using ForwardDiff

# INPUTS

quad1D = Inti.GaussLegendre(2)
x̂ = quad1D.nodes[1][1]
x̂ = SVector(x̂, x̂) # source point in reference coordinates
ori = 1  # element orientation

E = 210e9
ν = 0.3
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2ν))

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
n_rho = 5
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
x = el(x̂)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

# Kernel setup
op = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)
K_base = Inti.HyperSingularKernel(op)
K = GRD.SplitKernel(K_base)

K_polar = GRD.polar_kernel_fun(K_base, el, û, x̂, ori)

quad_rho = Inti.GaussLegendre(n_rho)

ref_domain = Inti.reference_domain(el)
ρ_max_fun = GRD.rho_fun(ref_domain, x̂)

decompo = Inti.polar_decomposition(ref_domain, x̂)

ℒ = GRD.laurents_coeffs(K, el, ori, û, x̂, GRD.AutoDiffExpansion())

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
		last_rel = norm(I_curr - I_prev) / max(norm(I_curr), eps())
		if last_rel < tol
			return n, last_rel
		end
		I_prev = I_curr
	end
	return nmax, last_rel
end

N = 10000
θs = range(0, stop = 2π, length = N)
fig = Figure()
## add an overall title
ax11 = Axis(fig[1, 1]; xlabel = "θ", ylabel = "G₁(θ)")

colors = [[:blue, :orange, :green], [:red, :purple, :brown], [:pink, :gray, :cyan]]
g1_ymin = Ref(Inf)
g1_ymax = Ref(-Inf)

for i in 1:3
	for j in 1:3
		g = [G₁(θ)[i, j] for θ in θs]
		local_min, local_max = extrema(g)
		g1_ymin[] = min(g1_ymin[], local_min)
		g1_ymax[] = max(g1_ymax[], local_max)
		lines!(ax11, θs, g, label = "G₁(θ)[$i,$j]", color = colors[i][j])
	end
end
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
	println("Secteur T=$T | tol=$target_rel_tol | nθ(G1)=$n1 (rel=$rel1) | nθ(G2)=$n2 (rel=$rel2)")
end

ax11.title = "G₁(θ) | tol=$(target_rel_tol)"
g1_ylabel = g1_ymax[] - 0.06 * (g1_ymax[] - g1_ymin[])

vlines!(ax11, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

for T in 1:4
	θa = θ_bounds[T]
	θb = θ_bounds[T + 1]
	θlabel = θa + 0.04 * (θb - θa)
	text!(ax11, θlabel, g1_ylabel; text = "T=$T: nG1=$(nθ_G1[T])", align = (:left, :top), fontsize = 10, color = :black)
end

ax21 = Axis(fig[2, 1]; xlabel = "θ", ylabel = "G₂(θ)")
g2_ymin = Ref(Inf)
g2_ymax = Ref(-Inf)
for i in 1:3
	for j in 1:3
		g = [G₂(θ)[i, j] for θ in θs]
		local_min, local_max = extrema(g)
		g2_ymin[] = min(g2_ymin[], local_min)
		g2_ymax[] = max(g2_ymax[], local_max)
		lines!(ax21, θs, g, label = "G₂(θ)[$i,$j]", color = colors[i][j])
	end
end

g2_ylabel = g2_ymax[] - 0.06 * (g2_ymax[] - g2_ymin[])

vlines!(ax21, [θ₁, θ₂, θ₃, θ₄]; color = :red, linestyle = :dash, label = "Sector boundaries")

for T in 1:4
	θa = θ_bounds[T]
	θb = θ_bounds[T + 1]
	θlabel = θa + 0.04 * (θb - θa)
	text!(ax21, θlabel, g2_ylabel; text = "T=$T: nG2=$(nθ_G2[T])", align = (:left, :top), fontsize = 10, color = :black)
end

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

window_1 = display(GLMakie.Screen(), fig)
# GLMakie.save("./dev/figures/navier/navier_hypersingular_angular_function.png", fig)
