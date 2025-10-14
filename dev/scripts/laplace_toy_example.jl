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

expected_I = -1 / (4π) * 5.749237

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)
# p = 1
# û = ξ -> Inti.lagrange_basis(typeof(el))(ξ)[p]
û = ξ -> 1.0

K = GRD.SplitLaplaceHypersingular

D = Dict{Symbol, Tuple{Function, Function}}()

maxeval = 10
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :analytical, name = :LaplaceHypersingular)
D[:analytical] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_1)
D[:semi_analytical_lvl_1] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_2, maxeval = maxeval, rtol = rtol, first_contract = first_contract, breaktol = breaktol, contract = contract, atol = atol)
D[:semi_analytical_lvl_2] = (F₋₂, F₋₁)

F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :full_richardson, maxeval = maxeval, rtol = rtol, first_contract = first_contract, breaktol = breaktol, contract = contract, atol = atol)
D[:full_richardson] = (F₋₂, F₋₁)

N = 1000
θs = range(0, 2π, length = N)

fig1 = Figure(; size = (1200, 800))
ax1 = Axis(fig1[1, 1]; xlabel = "θ", ylabel = "Laurent Coefficients", title = "Laplace Hypersingular Kernel Laurent Coefficients, Richardson max eval = $maxeval")

for (method, (F₋₂, F₋₁)) in D
	p = 2
	if method != :analytical
		error_F₋₂ = norm(D[:analytical][1].(θs) - F₋₂.(θs), p) / norm(D[:analytical][1].(θs), p)
		@info "Relative error (order $p) F₋₂ ($method vs analytical): $(maximum(error_F₋₂))"
		error_F₋₁ = norm(D[:analytical][2].(θs) - F₋₁.(θs), p) / norm(D[:analytical][2].(θs), p)
		@info "Relative error (order $p) F₋₁ ($method vs analytical): $(maximum(error_F₋₁))"
		lines!(ax1, θs, F₋₂.(θs); label = "F₋₂ $method (error = $(round(error_F₋₂, sigdigits=3)))", linewidth = 4)
		lines!(ax1, θs, F₋₁.(θs); label = "F₋₁ $method (error = $(round(error_F₋₁, sigdigits=3)))", linewidth = 4, linestyle = :dash)
	end
end

method = :analytical
F₋₂, F₋₁ = D[method]
lines!(ax1, θs, F₋₂.(θs); label = "F₋₂ $method", linewidth = 4)
lines!(ax1, θs, F₋₁.(θs); label = "F₋₁ $method", linewidth = 4, linestyle = :dash)

maxevals = 1:10

errors_F₋₂ = zeros(length(maxevals))
errors_F₋₁ = zeros(length(maxevals))

errors_G₋₂ = zeros(length(maxevals))
errors_G₋₁ = zeros(length(maxevals))

for (i, maxeval_) in enumerate(maxevals)
	F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :full_richardson, maxeval = maxeval_, rtol = rtol, first_contract = first_contract, breaktol = breaktol, contract = contract, atol = atol)
	error_F₋₂ = norm(D[:analytical][1].(θs) - F₋₂.(θs), 2) / norm(D[:analytical][1].(θs), 2)
	errors_F₋₂[i] = error_F₋₂
	error_F₋₁ = norm(D[:analytical][2].(θs) - F₋₁.(θs), 2) / norm(D[:analytical][2].(θs), 2)
	errors_F₋₁[i] = error_F₋₁
	G₋₂, G₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_analytical_lvl_2, maxeval = maxeval_, rtol = rtol, first_contract = first_contract, breaktol = breaktol, contract = contract, atol = atol)
	error_G₋₂ = norm(D[:analytical][1].(θs) - G₋₂.(θs), 2) / norm(D[:analytical][1].(θs), 2)
	error_G₋₁ = norm(D[:analytical][2].(θs) - G₋₁.(θs), 2) / norm(D[:analytical][2].(θs), 2)
	errors_G₋₂[i] = error_G₋₂
	errors_G₋₁[i] = error_G₋₁
	@info "maxeval = $maxeval_"
end

fig2 = Figure(; size = (1200, 800))
ax2 = Axis(
	fig2[1, 1];
	xlabel = "maxeval",
	ylabel = "Relative Error in norm 2",
	title = "Laplace Hypersingular Kernel Laurent Coefficients Error vs maxeval, Richardson first contract = $first_contract, contract = $contract, rtol = $rtol, breaktol = $breaktol, atol = $atol",
	yscale = log10,
)
lines!(ax2, maxevals, errors_F₋₂; label = "F₋₂ full_richardson", linewidth = 4)
lines!(ax2, maxevals, errors_F₋₁; label = "F₋₁ full_richardson", linewidth = 4)
lines!(ax2, maxevals, errors_G₋₁; label = "F₋₁ semi_analytical_lvl_2", linewidth = 4)

# horizontal lines for machine precision
hlines!(ax2, [cbrt(eps())]; label = "∛ϵ", color = :black, linestyle = :dashdot, linewidth = 2)

# minimum of each curve
min_error_F₋₂, min_index_F₋₂ = findmin(errors_F₋₂)
min_error_F₋₁, min_index_F₋₁ = findmin(errors_F₋₁)
min_error_G₋₁, min_index_G₋₁ = findmin(errors_G₋₁)
hlines!(ax2, [min_error_F₋₂]; label = "min F₋₂ full_richardson = $(round(min_error_F₋₂, sigdigits=3)) at maxeval = $(maxevals[min_index_F₋₂])", color = :blue, linewidth = 2)
hlines!(ax2, [min_error_F₋₁]; label = "min F₋₁ full_richardson = $(round(min_error_F₋₁, sigdigits=3)) at maxeval = $(maxevals[min_index_F₋₁])", color = :orange, linewidth = 2)
hlines!(ax2, [min_error_G₋₁]; label = "min F₋₁ semi_analytical_lvl_2 = $(round(min_error_G₋₁, sigdigits=3)) at maxeval = $(maxevals[min_index_G₋₁])", color = :green, linewidth = 2)

axislegend(ax2; position = :rb)

axislegend(ax1; position = :rt)
GLMakie.save("./dev/figures/laplace_hypersingular_laurent_coeffs_all_methods.png", fig1)
# GLMakie.save("./dev/figures/laplace_hypersingular_laurent_coeffs_error_vs_maxeval_first_contract_$(first_contract).png", fig2)

M = 10

method = :analytical

# boucle sur rho
n_theta = 100

n_rhos = 1:M
errors_rho = zeros(length(n_rhos))

for (i, n_rho) in enumerate(n_rhos)
	@info "n_rho = $n_rho"

	I = GRD.guiggiani_singular_integral(
		K,
		û,
		x̂,
		el,
		n_rho,
		n_theta;
		sorder = Val(-2),
		expansion = method,
		rtol = rtol,
		maxeval = maxeval,
		first_contract = first_contract,
		breaktol = breaktol,
		contract = contract,
		atol = atol,
	)
	error = abs(I - expected_I) / abs(expected_I)
	errors_rho[i] = error
end

# boucle sur theta

n_rho = 100

n_thetas = 1:M
errors_theta = zeros(length(n_thetas))

for (i, n_theta) in enumerate(n_thetas)
	@info "n_theta = $n_theta"

	I = GRD.guiggiani_singular_integral(
		K,
		û,
		x̂,
		el,
		n_rho,
		n_theta;
		sorder = Val(-2),
		expansion = method,
		maxeval = maxeval,
		rtol = rtol,
		first_contract = first_contract,
		breaktol = breaktol,
		contract = contract,
		atol = atol,
	)
	error = abs(I - expected_I) / abs(expected_I)
	errors_theta[i] = error
end

fig3 = Figure(; size = (1200, 800))
ax3 = Axis(
	fig3[1, 1];
	xlabel = "nombre de points de quadrature",
	ylabel = "Relative Error",
	title = "Laplace Hypersingular Integral quadrature error vs number of quadrature points, Richardson first contract = $first_contract, contract = $contract, rtol = 1e-9, breaktol = $breaktol, atol = $atol",
	yscale = log10,
)

lines!(ax3, n_rhos, errors_rho; label = "error vs n_rho, n_theta = $n_theta", linewidth = 4)
lines!(ax3, n_thetas, errors_theta; label = "error vs n_theta, n_rho = $n_rho", linewidth = 4)

axislegend(ax3, position = :rt)

save("./dev/figures/laplace_hypersingular_integral_error_vs_n_rho_n_theta_first_contract_$(first_contract)_method_$(method).png", fig3)

