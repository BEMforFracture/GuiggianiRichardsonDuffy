import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

# INPUTS

# material properties

μ = 1.0
λ = 1.0

x̂ = SVector(0.5, 0.5) # source point in reference coordinates

### Richardson extrapolation parameters
maxeval = 10
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

maxeval_in_loop = 100

# END INPUTS

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)
ref_domain = Inti.reference_domain(el)
# p = 1
# û = ξ -> Inti.lagrange_basis(typeof(el))(ξ)[p]
û = ξ -> 1.0

K = GRD.SplitElastostaticHypersingular

# Compute Laurent coefficients for each method - now returns a single function ℒ
D = Dict{Symbol, Function}()

D[:analytical] = GRD.laurents_coeffs(K, el, û, x̂; expansion = :analytical, name = :ElastostaticHypersingular, μ = μ, λ = λ)

D[:semi_richardson] = GRD.laurents_coeffs(K, el, û, x̂;
	expansion = :semi_richardson,
	kernel_kwargs = (μ = μ, λ = λ),
	richardson_kwargs = (
		maxeval = maxeval, rtol = rtol, atol = atol, contract = contract, first_contract = first_contract, breaktol = breaktol,
	),
)

D[:full_richardson] = GRD.laurents_coeffs(K, el, û, x̂;
	expansion = :full_richardson,
	kernel_kwargs = (μ = μ, λ = λ),
	richardson_kwargs = (
		maxeval = maxeval, rtol = rtol, atol = atol, contract = contract, first_contract = first_contract, breaktol = breaktol,
	),
)

D[:auto_diff] = GRD.laurents_coeffs(K, el, û, x̂;
	expansion = :auto_diff,
	kernel_kwargs = (μ = μ, λ = λ),
)

N = 1000
θs = range(0, 2π, length = N)

fig1 = Figure(; size = (1200, 800))
ax1 = Axis(fig1[1, 1]; xlabel = "θ", ylabel = "Laurent Coefficients", title = "Navier Hypersingular Kernel Laurent Coefficients, Richardson max eval = $maxeval")

for (method, ℒ) in D
	p = 2
	if method != :analytical
		vals_ref = [D[:analytical](θ) for θ in θs]
		vals_test = [ℒ(θ) for θ in θs]
		
		F₋₂_ref = [norm(v[1]) for v in vals_ref]
		F₋₂_test = [norm(v[1]) for v in vals_test]
		F₋₁_ref = [norm(v[2]) for v in vals_ref]
		F₋₁_test = [norm(v[2]) for v in vals_test]
		
		error_F₋₂ = norm(F₋₂_ref - F₋₂_test, p) / norm(F₋₂_ref, p)
		@info "Relative error (order $p) F₋₂ ($method vs analytical): $(maximum(error_F₋₂))"
		error_F₋₁ = norm(F₋₁_ref - F₋₁_test, p) / norm(F₋₁_ref, p)
		@info "Relative error (order $p) F₋₁ ($method vs analytical): $(maximum(error_F₋₁))"
		lines!(ax1, θs, F₋₂_test; label = "F₋₂ $method (error = $(round(error_F₋₂, sigdigits=3)))", linewidth = 4)
		lines!(ax1, θs, F₋₁_test; label = "F₋₁ $method (error = $(round(error_F₋₁, sigdigits=3)))", linewidth = 4, linestyle = :dash)
	end
end

method = :analytical
ℒ_ana = D[method]
vals_ana = [ℒ_ana(θ) for θ in θs]
F₋₂_ana = [norm(v[1]) for v in vals_ana]
F₋₁_ana = [norm(v[2]) for v in vals_ana]
lines!(ax1, θs, F₋₂_ana; label = "F₋₂ $method", linewidth = 4)
lines!(ax1, θs, F₋₁_ana; label = "F₋₁ $method", linewidth = 4, linestyle = :dash)

axislegend(ax1; position = :rt)

GLMakie.save("./dev/figures/navier_hypersingular_laurent_coeffs_all_methods.png", fig1)

maxevals = 1:maxeval_in_loop

errors_F₋₂ = zeros(length(maxevals))
errors_F₋₁ = zeros(length(maxevals))

errors_G₋₂ = zeros(length(maxevals))
errors_G₋₁ = zeros(length(maxevals))

ℒ_ana = D[:analytical]
vals_ana = [ℒ_ana(θ) for θ in θs]
F₋₂_ref = [norm(v[1]) for v in vals_ana]
F₋₁_ref = [norm(v[2]) for v in vals_ana]

for (i, maxeval_) in enumerate(maxevals)
	ℒ_full = GRD.laurents_coeffs(K, el, û, x̂;
		expansion = :full_richardson,
		kernel_kwargs = (μ = μ, λ = λ),
		richardson_kwargs = (
			maxeval = maxeval_, rtol = rtol, atol = atol, contract = contract, first_contract = first_contract, breaktol = breaktol,
		),
	)
	vals_full = [ℒ_full(θ) for θ in θs]
	F₋₂_full = [norm(v[1]) for v in vals_full]
	F₋₁_full = [norm(v[2]) for v in vals_full]
	
	error_F₋₂ = norm(F₋₂_ref - F₋₂_full, 2) / norm(F₋₂_ref, 2)
	errors_F₋₂[i] = error_F₋₂
	error_F₋₁ = norm(F₋₁_ref - F₋₁_full, 2) / norm(F₋₁_ref, 2)
	errors_F₋₁[i] = error_F₋₁
	
	ℒ_semi = GRD.laurents_coeffs(K, el, û, x̂;
		expansion = :semi_richardson,
		kernel_kwargs = (μ = μ, λ = λ),
		richardson_kwargs = (
			maxeval = maxeval_, rtol = rtol, atol = atol, contract = contract, first_contract = first_contract, breaktol = breaktol,
		),
	)
	vals_semi = [ℒ_semi(θ) for θ in θs]
	G₋₂_semi = [norm(v[1]) for v in vals_semi]
	G₋₁_semi = [norm(v[2]) for v in vals_semi]
	
	error_G₋₂ = norm(F₋₂_ref - G₋₂_semi, 2) / norm(F₋₂_ref, 2)
	error_G₋₁ = norm(F₋₁_ref - G₋₁_semi, 2) / norm(F₋₁_ref, 2)
	errors_G₋₂[i] = error_G₋₂
	errors_G₋₁[i] = error_G₋₁
	@info "maxeval = $maxeval_"
end

fig2 = Figure(; size = (1200, 800))
ax2 = Axis(
	fig2[1, 1];
	xlabel = "maxeval",
	ylabel = "Relative Error in norm 2",
	title = "Navier Hypersingular Kernel Laurent Coefficients Error vs maxeval, Richardson first contract = $first_contract, contract = $contract, rtol = $rtol, breaktol = $breaktol, atol = $atol",
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

GLMakie.save("./dev/figures/navier_hypersingular_laurent_coeffs_error_vs_maxeval_first_contract_$(first_contract).png", fig2)
