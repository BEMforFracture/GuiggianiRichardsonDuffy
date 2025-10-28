import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

# Configuration
x̂ = SVector(0.05, 0.05)  # Source point in reference coordinates

# Quadrature convergence test parameters
M_rho = 10  # Fixed n_rho when varying theta
M_theta = 50  # Fixed n_theta when varying rho
N_max_rho = 20  # Max n_rho to test
N_max_theta = 20  # Max n_theta to test

# Richardson parameters
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 8,
)

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

# Reference value
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim=3))
K = GRD.SplitKernel(K_base)

fig = Figure(; size = (1200, 800))

# boucle sur rho
n_theta = M_theta

n_rhos = 1:N_max_rho
errors_rho = zeros(length(n_rhos))

method = :full_richardson

ax4 = Axis(
	fig[1, 1];
	ylabel = "Relative Error",
	title = "$method, first contract = $first_contract, contract = $contract, rtol = $rtol, breaktol = $breaktol, atol = $atol, maxeval = $maxeval",
	yscale = log10,
)

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

n_rho = M_rho

n_thetas = 1:N_max_theta
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

lines!(ax4, n_rhos, errors_rho; label = "error vs n_rho, n_theta = $n_theta", linewidth = 4)
lines!(ax4, n_thetas, errors_theta; label = "error vs n_theta, n_rho = $n_rho", linewidth = 4)

method = :semi_richardson

ax5 = Axis(
	fig[2, 1];
	xlabel = "nombre de points de quadrature",
	ylabel = "Relative Error",
	title = "$method, first contract = $first_contract, contract = $contract, rtol = $rtol, breaktol = $breaktol, atol = $atol, maxeval = $maxeval",
	yscale = log10,
)

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
n_rho = M_rho
n_thetas = 1:N_max_theta
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

lines!(ax5, n_rhos, errors_rho; label = "error vs n_rho, n_theta = $n_theta", linewidth = 4)
lines!(ax5, n_thetas, errors_theta; label = "error vs n_theta, n_rho = $n_rho", linewidth = 4)

method = :analytical

ax6 = Axis(
	fig[2, 2];
	xlabel = "nombre de points de quadrature",
	title = "$method",
	yscale = log10,
)

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
	)
	error = abs(I - expected_I) / abs(expected_I)
	errors_rho[i] = error
end

# boucle sur theta

n_rho = M_rho
n_thetas = 1:N_max_theta
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
	)
	error = abs(I - expected_I) / abs(expected_I)
	errors_theta[i] = error
end

lines!(ax6, n_rhos, errors_rho; label = "error vs n_rho, n_theta = $n_theta", linewidth = 4)
lines!(ax6, n_thetas, errors_theta; label = "error vs n_theta, n_rho = $n_rho", linewidth = 4)

method = :auto_diff

ax7 = Axis(
	fig[1, 2];
	title = "$method",
	yscale = log10,
)

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
	)
	error = abs(I - expected_I) / abs(expected_I)
	errors_rho[i] = error
end

# boucle sur theta

n_rho = M_rho
n_thetas = 1:N_max_theta
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
	)

	error = abs(I - expected_I) / abs(expected_I)
	errors_theta[i] = error
end

lines!(ax7, n_rhos, errors_rho; label = "error vs n_rho, n_theta = $n_theta", linewidth = 4)
lines!(ax7, n_thetas, errors_theta; label = "error vs n_theta, n_rho = $n_rho", linewidth = 4)

axislegend(ax4, position = :rt)
axislegend(ax5, position = :rt)
axislegend(ax6, position = :rt)
axislegend(ax7, position = :rt)

# GLMakie.save("./dev/figures/laplace/laplace_hypersingular_integral_error_vs_n_rho_n_theta_all_methods.png", fig)
