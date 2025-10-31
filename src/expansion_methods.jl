function _create_laurent_coeffs_function(
	method::FullRichardsonExpansion,
	K::Inti.AbstractKernel,
	el::Inti.ReferenceInterpolant,
	û,
	x̂,
	ori,
)
	params = method.richardson_params
	s = Inti.singularity_order(K)
	sorder = Val(s + 1)

	# Pre-compute once
	SK = K isa SplitKernel ? K : SplitKernel(K)
	Kprod = (qx, qy) -> prod(SK(qx, qy))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂, ori)
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
	return function (θ)
		h = rho_max_fun(θ) * params.first_contract
		f = ρ -> K_polar(ρ, θ)
		return __laurents_coeff_full_richardson(f, h, sorder; kwargs_rich...)
	end
end

function _richardson_guard(h0; kwargs...)
	kwargs = Dict(kwargs)
	if (haskey(kwargs, :contract) && haskey(kwargs, :maxeval))
		x_cr = h0 * kwargs[:contract]^(kwargs[:maxeval])
		if x_cr < eps(Float64)
			@warn "Richardson extrapolation may be inaccurate or produces errors: final step size $x_cr is below machine epsilon."
		end
	end
end

function __laurents_coeff_full_richardson(f, h, ::Val{-2}; kwargs...)
	_richardson_guard(h; kwargs...)
	g = x -> x^2 * f(x)
	f₋₂, e₋₂ = extrapolate(h; kwargs...) do x
		return g(x)
	end
	f₋₁, e₋₁ = extrapolate(h; kwargs...) do x
		return x * f(x) - f₋₂ / x
	end
	return f₋₂, f₋₁
end

function __laurents_coeff_full_richardson(f, h, ::Val{-1}; kwargs...)
	_richardson_guard(h; kwargs...)
	g = ρ -> ρ * f(ρ)
	f₋₁, e₋₁ = extrapolate(h; kwargs...) do x
		return g(x)
	end
	return zero(f₋₁), f₋₁
end

function __laurents_coeff_full_richardson(f, h, ::Val{N}; kwargs...) where {N}
	_richardson_guard(h; kwargs...)
	if N > 0
		return 0.0, 0.0
	else
		throw(ArgumentError("order must be >= -2"))
	end
end

function _create_laurent_coeffs_function(
	method::SemiRichardsonExpansion,
	K::Inti.AbstractKernel,
	el::Inti.ReferenceInterpolant,
	û,
	x̂,
	ori,
)
	params = method.richardson_params
	s = Inti.singularity_order(K)
	sorder = Val(s + 1)

	# Pre-compute once
	SK = K isa SplitKernel ? K : SplitKernel(K)
	Kprod = (qx, qy) -> prod(SK(qx, qy))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂, ori)
	ref_domain = Inti.reference_domain(el)
	rho_max_fun = rho_fun(ref_domain, x̂)
	A = A_func(el, x̂)

	x = el(x̂)
	jac_x = Inti.jacobian(el, x̂)
	nx = Inti._normal(jac_x, ori)
	μ = Inti._integration_measure(jac_x)
	qx = (coords = x, normal = nx)

	kwargs_rich = (
		contract = params.contract,
		breaktol = params.breaktol,
		atol = params.atol,
		rtol = params.rtol,
		maxeval = params.maxeval,
	)

	# Return lightweight closure
	return function (θ)
		Â = A(θ) / norm(A(θ))
		_, K̂ = SK(qx, qx, Â)
		v = û(x̂)
		f_dom = map(v -> K̂ * v, v) * μ / norm(A(θ))^(-s)
		h = rho_max_fun(θ) * params.first_contract
		f = ρ -> K_polar(ρ, θ)
		return __laurents_coeff_semi_richardson(f, f_dom, h, sorder; kwargs_rich...)
	end
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

function _create_laurent_coeffs_function(
	method::AutoDiffExpansion,
	K::Inti.AbstractKernel,
	el::Inti.ReferenceInterpolant,
	û,
	x̂,
	ori,
)
	SK = K isa SplitKernel ? K : SplitKernel(K)
	s = Inti.singularity_order(K)
	S = s + 1
	sorder = Val(S)

	# Pre-compute once
	x = el(x̂)
	Dτ = Inti.jacobian(el, x̂)
	nx = Inti._normal(Dτ, ori)
	D²τ = Inti.hessian(el, x̂)
	qx = (coords = x, normal = nx)
	N = length(x)

	# Return lightweight closure
	return function (θ)
		function ℱ(ρ)
			uθ = u_func(θ)
			ŷ = x̂ + ρ * uθ
			jac_y = Inti.jacobian(el, ŷ)
			ny = Inti._normal(jac_y, ori)
			y = el(ŷ)
			qy = (coords = y, normal = ny)
			μ = Inti._integration_measure(jac_y)

			δ = ntuple(i -> transpose(uθ) * D²τ[i, :, :] * uθ, N) |> SVector
			A = Dτ * uθ + ρ / 2 * δ
			Â = A / norm(A)

			_, K̂ = SK(qx, qy, Â)

			v = û(ŷ)
			return map(v -> K̂ * v, v) * μ / norm(A)^(-S + 1)
		end

		return __laurents_coeff_auto_diff(ℱ, sorder)
	end
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

function _create_laurent_coeffs_function(
	method::AnalyticalExpansion,
	K::Inti.AbstractKernel,
	el::Inti.ReferenceInterpolant,
	û,
	x̂,
	ori,
)
	# Return lightweight closure - no pre-computation needed for analytical
	return θ -> _laurents_coeffs_closed_forms(K, θ, x̂, el, û)
end
