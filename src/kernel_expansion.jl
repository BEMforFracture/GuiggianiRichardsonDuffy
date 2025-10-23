function _laurents_coeff_full_richardson(K, el::Inti.ReferenceInterpolant, û, x̂, kwargs_kernel::NamedTuple, kwargs_rich::NamedTuple)
	Kprod = (qx, qy) -> prod(K(qx, qy; kwargs_kernel...))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	ref_domain = Inti.reference_domain(el)
	rho_max_fun = rho_fun(ref_domain, x̂)
	h_user = haskey(kwargs_rich, :first_contract) ? kwargs_rich.first_contract : 1e-2
	ratio = haskey(kwargs_rich, :contract) ? kwargs_rich.contract : 0.5
	breaktol = haskey(kwargs_rich, :breaktol) ? kwargs_rich.breaktol : 2
	maxeval = haskey(kwargs_rich, :maxeval) ? kwargs_rich.maxeval : typemax(Int)
	atol = haskey(kwargs_rich, :atol) ? kwargs_rich.atol : 0.0
	rtol = haskey(kwargs_rich, :rtol) ? kwargs_rich.rtol : (atol > 0 ? 0.0 : sqrt(eps()))
	@memoize function ℒ(θ)
		h = rho_max_fun(θ) * h_user
		g = ρ -> ρ^2 * K_polar(ρ, θ)
		f₋₂, e₋₂ = extrapolate(h; contract = ratio, x0 = 0, atol = atol, rtol = rtol, maxeval = maxeval, breaktol = breaktol) do x
			return g(x)
		end
		f₋₁, e₋₁ = extrapolate(h; contract = ratio, x0 = 0, atol = atol, rtol = rtol, maxeval = maxeval, breaktol = breaktol) do x
			return x * K_polar(x, θ) - f₋₂ / x
		end
		return f₋₂, f₋₁
	end
	return ℒ
end

function polar_kernel_fun_normalized(K, el::Inti.ReferenceInterpolant, û, x̂; kwargs...)
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
		return K̂ * μ * û(ŷ) / norm(A)^3
	end
	return ℱ
end

function _laurents_coeff_auto_diff(K, el::Inti.ReferenceInterpolant, û, x̂; kwargs...)
	ℱ = polar_kernel_fun_normalized(K, el, û, x̂; kwargs...)
	
	# Memoïze les coefficients pour éviter de recalculer ForwardDiff à chaque appel
	@memoize function ℒ(θ)
		f₋₂ = ℱ(0, θ)
		f₋₁ = ForwardDiff.derivative(ρ -> ℱ(ρ, θ), 0.0)
		return f₋₂, f₋₁
	end
	
	return ℒ
end

function _laurents_coeff_semi_richardson(K, el::Inti.ReferenceInterpolant, û, x̂, kwargs_kernel::NamedTuple, kwargs_rich::NamedTuple)
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
	# Précalculer K_polar une seule fois au lieu de le recréer à chaque appel à ℒ(θ)
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	h_user = haskey(kwargs_rich, :first_contract) ? kwargs_rich.first_contract : 1e-2
	ratio = haskey(kwargs_rich, :contract) ? kwargs_rich.contract : 0.5
	breaktol = haskey(kwargs_rich, :breaktol) ? kwargs_rich.breaktol : 2
	maxeval = haskey(kwargs_rich, :maxeval) ? kwargs_rich.maxeval : typemax(Int)
	atol = haskey(kwargs_rich, :atol) ? kwargs_rich.atol : 0.0
	rtol = haskey(kwargs_rich, :rtol) ? kwargs_rich.rtol : (atol > 0 ? 0.0 : sqrt(eps()))
	@memoize function ℒ(θ)
		Â = A(θ) / norm(A(θ))
		_, K̂ = K(qx, qx, Â; kwargs_kernel...)
		F₋₂ = K̂ * μ * û(x̂) / norm(A(θ))^3
		h = rho_max_fun(θ) * h_user
		F₋₁, e₋₁ = extrapolate(h; contract = ratio, x0 = 0, atol = atol, rtol = rtol, maxeval = maxeval, breaktol = breaktol) do x
			return x * K_polar(x, θ) - F₋₂ / x
		end
		return F₋₂, F₋₁
	end
	return ℒ
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
