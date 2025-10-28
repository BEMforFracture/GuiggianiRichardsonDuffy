# ==============================================================================
# Direct implementations without LaurentExpander (lightweight, like main branch)
# ==============================================================================

function _create_laurent_coeffs_function(
	method::FullRichardsonExpansion,
	K::Inti.AbstractKernel,
	el::Inti.ReferenceInterpolant,
	û,
	x̂,
)
	params = method.richardson_params
	s = Inti.singularity_order(K)
	sorder = Val(s + 1)
	
	# Pre-compute once
	SK = K isa SplitKernel ? K : SplitKernel(K)
	Kprod = (qx, qy) -> prod(SK(qx, qy))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	ref_domain = Inti.reference_domain(el)
	rho_max_fun = rho_fun(ref_domain, x̂)
	
	kwargs_rich = (
		contract = params.contract,
		breaktol = params.breaktol,
		atol = params.atol,
		rtol = params.rtol,
		maxeval = params.maxeval,
	)
	
	# Return lightweight closure
	return function(θ)
		h = rho_max_fun(θ) * params.first_contract
		f = ρ -> K_polar(ρ, θ)
		return __laurents_coeff_full_richardson(f, h, sorder; kwargs_rich...)
	end
end

function __laurents_coeff_full_richardson(f, h, ::Val{-2}; kwargs...)
	g = ρ -> ρ^2 * f(ρ)
	f₋₂, e₋₂ = extrapolate(h; kwargs...) do x
		return g(x)
	end
	f₋₁, e₋₁ = extrapolate(h; kwargs...) do x
		return x * f(x) - f₋₂ / x
	end
	return f₋₂, f₋₁
end

function __laurents_coeff_full_richardson(f, h, ::Val{-1}; kwargs...)
	g = ρ -> ρ * f(ρ)
	f₋₁, e₋₁ = extrapolate(h; kwargs...) do x
		return g(x)
	end
	return zero(f₋₁), f₋₁
end

function __laurents_coeff_full_richardson(f, h, ::Val{N}; kwargs...) where {N}
	if N > 0
		return 0.0, 0.0
	else
		throw(ArgumentError("order must be >= -2"))
	end
end

function compute_coefficients(
	expander::LaurentExpander{SemiRichardsonExpansion},
	θ,
)
	params = expander.method.richardson_params

	# Compute A function (still needs to be done per theta)
	A = A_func(expander.reference_element, expander.source_point)
	
	# Create qx
	qx = (coords = expander.x, normal = expander.nx)
	
	# Use pre-computed SplitKernel
	SK = expander.kernel isa SplitKernel ? expander.kernel : SplitKernel(expander.kernel)

	Â = A(θ) / norm(A(θ))
	_, K̂ = SK(qx, qx, Â)

	s = Inti.singularity_order(expander.kernel)
	sorder = Val(s + 1)

	f_dom = K̂ * expander.μ * expander.û(expander.source_point) / norm(A(θ))^(-s)

	# Use pre-computed closures
	h = expander.rho_max_fun(θ) * params.first_contract
	f = ρ -> expander.K_polar(ρ, θ)

	kwargs_rich = (
		contract = params.contract,
		breaktol = params.breaktol,
		atol = params.atol,
		rtol = params.rtol,
		maxeval = params.maxeval,
	)

	return __laurents_coeff_semi_richardson(f, f_dom, h, sorder; kwargs_rich...)
end

function __laurents_coeff_semi_richardson(f, f_dom, h, ::Val{-2}; kwargs...)
	f₋₁, e₋₁ = extrapolate(h; kwargs...) do x
		return x * f(x) - f_dom / x
	end
	return f_dom, f₋₁
end

function __laurents_coeff_semi_richardson(f, f_dom, h, ::Val{-1}; kwargs...)
	return zero(f_dom), f_dom
end

function __laurents_coeff_semi_richardson(f, f_dom, h, ::Val{N}; kwargs...) where {N}
	if N > 0
		return 0.0, 0.0
	else
		throw(ArgumentError("order must be >= -2"))
	end
end

function compute_coefficients(
	expander::LaurentExpander{AutoDiffExpansion, Tk, T, N, Nd, D, Tu},
	θ,
) where {Tk, T, N, Nd, D, Tu}
	qx = (coords = expander.x, normal = expander.nx)

	s = Inti.singularity_order(expander.kernel)
	S = s + 1

	function ℱ(ρ)
		uθ = u_func(θ)
		ŷ = expander.source_point + ρ * uθ
		jac_y = Inti.jacobian(expander.reference_element, ŷ)
		ori = 1
		ny = Inti._normal(jac_y, ori)
		y = expander.reference_element(ŷ)
		qy = (coords = y, normal = ny)
		μ = Inti._integration_measure(jac_y)

		# Calculer A (vecteur tangent)
		δ = ntuple(i -> transpose(uθ) * expander.D²τ[i, :, :] * uθ, N) |> SVector
		A = expander.Dτ * uθ + ρ / 2 * δ
		Â = A / norm(A)

		# Utiliser le split kernel
		_, K̂ = expander.kernel(qx, qy, Â)

		return K̂ * μ * expander.û(ŷ) / norm(A)^(-S + 1)
	end

	sorder = Val(S)

	return __laurents_coeff_auto_diff(ℱ, sorder)
end

function __laurents_coeff_auto_diff(f, ::Val{-2})
	f₋₂ = f(0.0)
	f₋₁ = ForwardDiff.derivative(f, 0.0)
	return f₋₂, f₋₁
end

function __laurents_coeff_auto_diff(f, ::Val{-1})
	f₋₁ = f(0.0)
	return zero(f₋₁), f₋₁
end

function __laurents_coeff_auto_diff(f, ::Val{N}) where {N}
	if N > 0
		return 0.0, 0.0
	else
		throw(ArgumentError("order must be >= -2"))
	end
end

function compute_coefficients(
	expander::LaurentExpander{AnalyticalExpansion, Tk, T, N, Nd, D, Tu},
	θ,
) where {Tk, T, N, Nd, D, Tu}
	return _laurents_coeffs_closed_forms(
		expander.kernel,
		θ,
		expander.source_point,
		expander.reference_element,
		expander.û,
	)
end
