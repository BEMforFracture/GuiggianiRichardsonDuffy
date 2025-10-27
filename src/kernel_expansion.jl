function _laurents_coeff_full_richardson(K, el::Inti.ReferenceInterpolant, û, x̂, kwargs_kernel::NamedTuple, kwargs_rich::NamedTuple; sorder::Val{S} = Val(-3), first_contract = 1e-2) where {S}
	Kprod = (qx, qy) -> prod(K(qx, qy; kwargs_kernel...))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	ref_domain = Inti.reference_domain(el)
	rho_max_fun = rho_fun(ref_domain, x̂)
	kwargs_rich = (; (k => v for (k, v) in pairs(kwargs_rich) if k != :first_contract)...)
	@memoize function ℒ(θ)
		h = rho_max_fun(θ) * first_contract
		f = ρ -> K_polar(ρ, θ)
		return __laurents_coeff_full_richardson(f, h, sorder; kwargs_rich...)
	end
	return ℒ
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

function polar_kernel_fun_normalized(K, el::Inti.ReferenceInterpolant, û, x̂; sorder::Val{S} = Val(-2), kwargs...) where {S}
	x = el(x̂)
	jac_x = Inti.jacobian(el, x̂)
	ori = 1
	nx = Inti._normal(jac_x, ori)
	qx = (coords = x, normal = nx)
	Dτ = Inti.jacobian(el, x̂)
	D²τ = Inti.hessian(el, x̂)
	function ℱ(ρ, θ)
		uθ = u_func(θ)
		ŷ = x̂ + ρ * uθ
		jac_y = Inti.jacobian(el, ŷ)
		ori = 1
		ny = Inti._normal(jac_y, ori)
		y = el(ŷ)
		qy = (coords = y, normal = ny)
		μ = Inti._integration_measure(jac_y)
		δ = ntuple(i -> transpose(uθ) * D²τ[i, :, :] * uθ, 3) |> SVector
		A = Dτ * uθ + ρ / 2 * δ
		Â = A / norm(A)
		_, K̂ = K(qx, qy, Â; kwargs...)
		return K̂ * μ * û(ŷ) / norm(A)^(-S + 1)
	end
	return ℱ
end

function _laurents_coeff_auto_diff(args...; kwargs...)
	ℱ = polar_kernel_fun_normalized(args...; kwargs...)

	sorder = get(kwargs, :sorder, Val(-2))

	# Memoïze les coefficients pour éviter de recalculer ForwardDiff à chaque appel
	@memoize function ℒ(θ)
		f₋₂, f₋₁ = __laurents_coeff_auto_diff(ρ -> ℱ(ρ, θ), sorder)
		return f₋₂, f₋₁
	end

	return ℒ
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

function _laurents_coeff_semi_richardson(K, el::Inti.ReferenceInterpolant, û, x̂, kwargs_kernel::NamedTuple, kwargs_rich::NamedTuple; sorder::Val{S} = Val(-2), first_contract = 1e-2) where {S}
	ref_domain = Inti.reference_domain(el)
	rho_max_fun = rho_fun(ref_domain, x̂)
	A = A_func(el, x̂)
	x = el(x̂)
	jac_x = Inti.jacobian(el, x̂)
	ori = 1
	nx = Inti._normal(jac_x, ori)
	qx = (coords = x, normal = nx)
	μ = Inti._integration_measure(jac_x)
	Kprod = (qx, qy) -> prod(K(qx, qy; kwargs_kernel...))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	@memoize function ℒ(θ)
		Â = A(θ) / norm(A(θ))
		_, K̂ = K(qx, qx, Â; kwargs_kernel...)
		f_dom = K̂ * μ * û(x̂) / norm(A(θ))^(-S + 1)
		h = rho_max_fun(θ) * first_contract
		f₋₂, f₋₁ = __laurents_coeff_semi_richardson(ρ -> K_polar(ρ, θ), f_dom, h, sorder; kwargs_rich...)
		return f₋₂, f₋₁
	end
	return ℒ
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

function _laurents_coeff_analytical(el::Inti.ReferenceInterpolant, û, x̂, arg...; name = :LaplaceHypersingular, kwargs...)
	if name == :LaplaceHypersingular
		return θ -> _laplace_hypersingular_closed_form_coeffs(θ, x̂, el, û, arg...; kwargs...)
	elseif name == :ElastostaticHypersingular
		return θ -> _elastostatic_hypersingular_closed_form_coeffs(θ, x̂, el, û, arg...; kwargs...)
	else
		error("Analytical laurent coefficients for kernel $(name) are not implemented. Available kernels are : $ANALYTICAL_KERNELS")
	end
end
