import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

# Configuration
x̂ = SVector(0.1, 0.1)  # Source point in reference coordinates

# Richardson extrapolation parameters for fixed method comparison
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 10,
)

# Range of maxeval values to test convergence
maxeval_in_loop = 100

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
K = GRD.SplitKernel(K_base)

# Compute Laurent coefficients for each method
methods = [
	("Analytical", GRD.AnalyticalExpansion()),
	("AutoDiff", GRD.AutoDiffExpansion()),
	("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params)),
	("FullRichardson", GRD.FullRichardsonExpansion(rich_params)),
]

D = Dict{String, Function}()

for (method_name, method) in methods
	# Use appropriate kernel
	K_to_use = (method isa GRD.AnalyticalExpansion) ? K_base : K
	D[method_name] = GRD.laurents_coeffs(K_to_use, el, û, x̂, method)
end

N = 1000
θs = range(0, 2π, length = N)

fig1 = Figure(; size = (1200, 800))
ax1 = Axis(fig1[1, 1]; xlabel = "θ", ylabel = "Laurent Coefficients",
	title = "Laplace Hypersingular Kernel Laurent Coefficients, Richardson maxeval = $(rich_params.maxeval)")

# Get analytical reference
ℒ_ana = D["Analytical"]
vals_analytical = [ℒ_ana(θ) for θ in θs]
F₋₂_ref = [v[1] for v in vals_analytical]
F₋₁_ref = [v[2] for v in vals_analytical]

# Plot all methods
for (method_name, ℒ) in D
	p = 2
	if method_name != "Analytical"
		# Compute values for comparison
		vals_test = [ℒ(θ) for θ in θs]
		F₋₂_test = [v[1] for v in vals_test]
		F₋₁_test = [v[2] for v in vals_test]

		# Compute errors
		error_F₋₂ = norm(F₋₂_ref - F₋₂_test, p) / norm(F₋₂_ref, p)
		@info "Relative error (order $p) F₋₂ ($method_name vs analytical): $(error_F₋₂)"
		error_F₋₁ = norm(F₋₁_ref - F₋₁_test, p) / norm(F₋₁_ref, p)
		@info "Relative error (order $p) F₋₁ ($method_name vs analytical): $(error_F₋₁)"

		lines!(ax1, θs, F₋₂_test; label = "F₋₂ $method_name (error = $(round(error_F₋₂, sigdigits=3)))", linewidth = 4)
		lines!(ax1, θs, F₋₁_test; label = "F₋₁ $method_name (error = $(round(error_F₋₁, sigdigits=3)))", linewidth = 4, linestyle = :dash)
	end
end

# Plot analytical reference
lines!(ax1, θs, F₋₂_ref; label = "F₋₂ Analytical", linewidth = 4)
lines!(ax1, θs, F₋₁_ref; label = "F₋₁ Analytical", linewidth = 4, linestyle = :dash)

axislegend(ax1; position = :rt)
GLMakie.save("./dev/figures/laplace/laplace_hypersingular_laurent_coeffs_all_methods.png", fig1)

maxevals = 1:maxeval_in_loop

errors_F₋₂ = zeros(length(maxevals))
errors_F₋₁ = zeros(length(maxevals))

errors_G₋₂ = zeros(length(maxevals))
errors_G₋₁ = zeros(length(maxevals))

# Get analytical reference once
vals_ref = [D["Analytical"](θ) for θ in θs]
F₋₂_ref = [v[1] for v in vals_ref]
F₋₁_ref = [v[2] for v in vals_ref]

for (i, maxeval_) in enumerate(maxevals)
	@info "Testing maxeval = $maxeval_"

	# Create params with varying maxeval
	params_var = GRD.RichardsonParams(
		first_contract = rich_params.first_contract,
		contract = rich_params.contract,
		breaktol = rich_params.breaktol,
		atol = rich_params.atol,
		rtol = rich_params.rtol,
		maxeval = maxeval_,
	)

	# Test FullRichardson
	ℒ_full = GRD.laurents_coeffs(K, el, û, x̂, GRD.FullRichardsonExpansion(params_var))
	vals_full = [ℒ_full(θ) for θ in θs]
	F₋₂_full = [v[1] for v in vals_full]
	F₋₁_full = [v[2] for v in vals_full]

	error_F₋₂ = norm(F₋₂_ref - F₋₂_full, 2) / norm(F₋₂_ref, 2)
	errors_F₋₂[i] = error_F₋₂
	error_F₋₁ = norm(F₋₁_ref - F₋₁_full, 2) / norm(F₋₁_ref, 2)
	errors_F₋₁[i] = error_F₋₁

	# Test SemiRichardson
	ℒ_semi = GRD.laurents_coeffs(K, el, û, x̂, GRD.SemiRichardsonExpansion(params_var))
	vals_semi = [ℒ_semi(θ) for θ in θs]
	G₋₂_semi = [v[1] for v in vals_semi]
	G₋₁_semi = [v[2] for v in vals_semi]

	error_G₋₂ = norm(F₋₂_ref - G₋₂_semi, 2) / norm(F₋₂_ref, 2)
	errors_G₋₂[i] = error_G₋₂
	error_G₋₁ = norm(F₋₁_ref - G₋₁_semi, 2) / norm(F₋₁_ref, 2)
	errors_G₋₁[i] = error_G₋₁
end

fig2 = Figure(; size = (1200, 800))
ax2 = Axis(
	fig2[1, 1];
	xlabel = "maxeval",
	ylabel = "Relative Error in norm 2",
	title = "Laplace Hypersingular Kernel Laurent Coefficients Error vs maxeval\nfirst_contract = $(rich_params.first_contract), contract = $(rich_params.contract), rtol = $(rich_params.rtol)",
	yscale = log10,
)
lines!(ax2, maxevals, errors_F₋₂; label = "F₋₂ full_richardson", linewidth = 4)
lines!(ax2, maxevals, errors_F₋₁; label = "F₋₁ full_richardson", linewidth = 4)
lines!(ax2, maxevals, errors_G₋₁; label = "F₋₁ semi_richardson", linewidth = 4)

# horizontal lines for machine precision
hlines!(ax2, [cbrt(eps())]; label = "∛ϵ", color = :black, linestyle = :dashdot, linewidth = 2)

# minimum of each curve
min_error_F₋₂, min_index_F₋₂ = findmin(errors_F₋₂)
min_error_F₋₁, min_index_F₋₁ = findmin(errors_F₋₁)
min_error_G₋₁, min_index_G₋₁ = findmin(errors_G₋₁)
hlines!(ax2, [min_error_F₋₂]; label = "min F₋₂ full_richardson = $(round(min_error_F₋₂, sigdigits=3)) at maxeval = $(maxevals[min_index_F₋₂])", color = :blue, linewidth = 2)
hlines!(ax2, [min_error_F₋₁]; label = "min F₋₁ full_richardson = $(round(min_error_F₋₁, sigdigits=3)) at maxeval = $(maxevals[min_index_F₋₁])", color = :orange, linewidth = 2)
hlines!(ax2, [min_error_G₋₁]; label = "min F₋₁ semi_richardson = $(round(min_error_G₋₁, sigdigits=3)) at maxeval = $(maxevals[min_index_G₋₁])", color = :green, linewidth = 2)

axislegend(ax2; position = :rb)

GLMakie.save("./dev/figures/laplace/laplace_hypersingular_laurent_coeffs_error_vs_maxeval_first_contract_$(first_contract).png", fig2)
