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
	maxeval = 4,
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
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
K = GRD.SplitKernel(K_base)

# Methods to test
methods = [
	("FullRichardson", GRD.FullRichardsonExpansion(rich_params), K),
	("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params), K),
	("Analytical", GRD.AnalyticalExpansion(), K_base),
	("AutoDiff", GRD.AutoDiffExpansion(), K),
]

fig = Figure(; size = (1200, 800))

for (plot_idx, (method_name, method, K_to_use)) in enumerate(methods)
	@info "Testing method: $method_name"

	# Determine subplot position
	row = div(plot_idx - 1, 2) + 1
	col = mod(plot_idx - 1, 2) + 1

	title_str = if method isa Union{GRD.FullRichardsonExpansion, GRD.SemiRichardsonExpansion}
		"$method_name (maxeval = $(rich_params.maxeval))"
	else
		method_name
	end

	ax = Axis(
		fig[row, col];
		xlabel = row == 2 ? "Number of quadrature points" : "",
		ylabel = "Relative Error",
		title = title_str,
		yscale = log10,
	)

	# Test varying n_rho (fixed n_theta)
	n_theta = M_theta
	quad_theta = Inti.GaussLegendre(n_theta)
	n_rhos = 1:N_max_rho
	errors_rho = zeros(length(n_rhos))

	for (i, n_rho) in enumerate(n_rhos)
		quad_rho = Inti.GaussLegendre(n_rho)
		I = GRD.guiggiani_singular_integral(
			K_base, û, x̂, el, ori, quad_rho, quad_theta, method,
		)
		error = abs(I - expected_I) / abs(expected_I)
		errors_rho[i] = error
	end

	# Test varying n_theta (fixed n_rho)
	n_rho = M_rho
	quad_rho = Inti.GaussLegendre(n_rho)
	n_thetas = 1:N_max_theta
	errors_theta = zeros(length(n_thetas))

	for (i, n_theta) in enumerate(n_thetas)
		quad_theta = Inti.GaussLegendre(n_theta)
		I = GRD.guiggiani_singular_integral(
			K_base, û, x̂, el, ori, quad_rho, quad_theta, method,
		)
		error = abs(I - expected_I) / abs(expected_I)
		errors_theta[i] = error
	end

	lines!(ax, n_rhos, errors_rho; label = "error vs n_rho (n_theta = $M_theta)", linewidth = 3)
	lines!(ax, n_thetas, errors_theta; label = "error vs n_theta (n_rho = $M_rho)", linewidth = 3)

	axislegend(ax, position = :rt)
end

# display(fig)
# GLMakie.save("./dev/figures/laplace/laplace_hypersingular_quadrature_convergence.png", fig)
