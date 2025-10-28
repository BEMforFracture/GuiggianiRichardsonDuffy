import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

# Configuration
x̂ = SVector(0.5, 0.5)

# Material properties
μ = 1.0
λ = 1.0

# Richardson parameters for display
maxeval_display = 10
rich_params_display = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = maxeval_display,
)

# Richardson parameters for convergence test
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
x = el(x̂)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

# Kernel setup
op = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)
K_base = Inti.HyperSingularKernel(op)
K = GRD.SplitKernel(K_base)

# Compute Laurent coefficients for each method
D = Dict{String, Function}()

D["Analytical"] = GRD.laurents_coeffs(K_base, el, û, x̂, GRD.AnalyticalExpansion())
D["AutoDiff"] = GRD.laurents_coeffs(K, el, û, x̂, GRD.AutoDiffExpansion())
D["SemiRichardson"] = GRD.laurents_coeffs(K, el, û, x̂, GRD.SemiRichardsonExpansion(rich_params_display))
D["FullRichardson"] = GRD.laurents_coeffs(K, el, û, x̂, GRD.FullRichardsonExpansion(rich_params_display))

# Angular samples
N = 1000
θs = range(0, 2π; length = N)

# Plot all methods
fig1 = Figure(; size = (1200, 800))
ax1 = Axis(fig1[1, 1];
	xlabel = "θ",
	ylabel = "Laurent Coefficients",
	title = "Navier Hypersingular Kernel Laurent Coefficients, Richardson maxeval = $maxeval_display",
)

for (method, ℒ) in D
	p = 2
	if method != "Analytical"
		vals_ref = [D["Analytical"](θ) for θ in θs]
		vals_test = [ℒ(θ) for θ in θs]

		F₋₂_ref = [norm(v[1]) for v in vals_ref]
		F₋₂_test = [norm(v[1]) for v in vals_test]
		F₋₁_ref = [norm(v[2]) for v in vals_ref]
		F₋₁_test = [norm(v[2]) for v in vals_test]

		error_F₋₂ = norm(F₋₂_ref - F₋₂_test, p) / norm(F₋₂_ref, p)
		@info "Relative error (order $p) F₋₂ ($method vs analytical): $(error_F₋₂)"
		error_F₋₁ = norm(F₋₁_ref - F₋₁_test, p) / norm(F₋₁_ref, p)
		@info "Relative error (order $p) F₋₁ ($method vs analytical): $(error_F₋₁)"

		lines!(ax1, θs, F₋₂_test;
			label = "F₋₂ $method (error = $(round(error_F₋₂, sigdigits=3)))",
			linewidth = 4,
		)
		lines!(ax1, θs, F₋₁_test;
			label = "F₋₁ $method (error = $(round(error_F₋₁, sigdigits=3)))",
			linewidth = 4,
			linestyle = :dash,
		)
	end
end

# Plot analytical reference
method = "Analytical"
ℒ_ana = D[method]
vals_ana = [ℒ_ana(θ) for θ in θs]
F₋₂_ana = [norm(v[1]) for v in vals_ana]
F₋₁_ana = [norm(v[2]) for v in vals_ana]
lines!(ax1, θs, F₋₂_ana; label = "F₋₂ $method", linewidth = 4)
lines!(ax1, θs, F₋₁_ana; label = "F₋₁ $method", linewidth = 4, linestyle = :dash)

axislegend(ax1; position = :rt)

# GLMakie.save("./dev/figures/navier/navier_hypersingular_laurent_coeffs_all_methods.png", fig1)

# Convergence test
maxevals = 1:maxeval_in_loop

errors_F₋₂ = zeros(length(maxevals))
errors_F₋₁ = zeros(length(maxevals))
errors_G₋₂ = zeros(length(maxevals))
errors_G₋₁ = zeros(length(maxevals))

# Reference values
ℒ_ana = D["Analytical"]
vals_ana = [ℒ_ana(θ) for θ in θs]
F₋₂_ref = [norm(v[1]) for v in vals_ana]
F₋₁_ref = [norm(v[2]) for v in vals_ana]

@info "Computing convergence for maxeval = 1:$maxeval_in_loop..."

for (i, maxeval_) in enumerate(maxevals)
	rich_params_test = GRD.RichardsonParams(
		first_contract = 1e-2,
		contract = 0.5,
		breaktol = Inf,
		atol = 0.0,
		rtol = 0.0,
		maxeval = maxeval_,
	)

	# Full Richardson
	ℒ_full = GRD.laurents_coeffs(K, el, û, x̂, GRD.FullRichardsonExpansion(rich_params_test))
	vals_full = [ℒ_full(θ) for θ in θs]
	F₋₂_full = [norm(v[1]) for v in vals_full]
	F₋₁_full = [norm(v[2]) for v in vals_full]

	error_F₋₂ = norm(F₋₂_ref - F₋₂_full, 2) / norm(F₋₂_ref, 2)
	errors_F₋₂[i] = error_F₋₂
	error_F₋₁ = norm(F₋₁_ref - F₋₁_full, 2) / norm(F₋₁_ref, 2)
	errors_F₋₁[i] = error_F₋₁

	# Semi Richardson
	ℒ_semi = GRD.laurents_coeffs(K, el, û, x̂, GRD.SemiRichardsonExpansion(rich_params_test))
	vals_semi = [ℒ_semi(θ) for θ in θs]
	G₋₂_semi = [norm(v[1]) for v in vals_semi]
	G₋₁_semi = [norm(v[2]) for v in vals_semi]

	error_G₋₂ = norm(F₋₂_ref - G₋₂_semi, 2) / norm(F₋₂_ref, 2)
	error_G₋₁ = norm(F₋₁_ref - G₋₁_semi, 2) / norm(F₋₁_ref, 2)
	errors_G₋₂[i] = error_G₋₂
	errors_G₋₁[i] = error_G₋₁

	if maxeval_ % 10 == 0
		@info "Progress: maxeval = $maxeval_"
	end
end

@info "Convergence computation complete. Creating plot..."

# Plot convergence
fig2 = Figure(; size = (1200, 800))
ax2 = Axis(
	fig2[1, 1];
	xlabel = "maxeval",
	ylabel = "Relative Error in norm 2",
	title = "Navier Hypersingular Kernel Laurent Coefficients Error vs maxeval\nRichardson first_contract = $(rich_params_display.first_contract), contract = $(rich_params_display.contract)",
	yscale = log10,
)

lines!(ax2, maxevals, errors_F₋₂; label = "F₋₂ FullRichardson", linewidth = 4)
lines!(ax2, maxevals, errors_F₋₁; label = "F₋₁ FullRichardson", linewidth = 4)
lines!(ax2, maxevals, errors_G₋₁; label = "F₋₁ SemiRichardson", linewidth = 4)

# Machine precision reference
hlines!(ax2, [cbrt(eps())]; label = "∛ϵ", color = :black, linestyle = :dashdot, linewidth = 2)

# Minimum of each curve
min_error_F₋₂, min_index_F₋₂ = findmin(errors_F₋₂)
min_error_F₋₁, min_index_F₋₁ = findmin(errors_F₋₁)
min_error_G₋₁, min_index_G₋₁ = findmin(errors_G₋₁)

hlines!(ax2, [min_error_F₋₂];
	label = "min F₋₂ FullRichardson = $(round(min_error_F₋₂, sigdigits=3)) at maxeval = $(maxevals[min_index_F₋₂])",
	color = :blue,
	linewidth = 2,
)
hlines!(ax2, [min_error_F₋₁];
	label = "min F₋₁ FullRichardson = $(round(min_error_F₋₁, sigdigits=3)) at maxeval = $(maxevals[min_index_F₋₁])",
	color = :orange,
	linewidth = 2,
)
hlines!(ax2, [min_error_G₋₁];
	label = "min F₋₁ SemiRichardson = $(round(min_error_G₋₁, sigdigits=3)) at maxeval = $(maxevals[min_index_G₋₁])",
	color = :green,
	linewidth = 2,
)

axislegend(ax2; position = :rb)

# GLMakie.save("./dev/figures/navier/navier_hypersingular_laurent_coeffs_error_vs_maxeval.png", fig2)
