#= Closed forme formulas for laurent's coefficients within Guiggiani algorithm =#

function LaplaceAdjointDoubleLayerClosedFormCoeffs()
	notimplemented()
end

@memoize function _laplace_hypersingular_closed_form_coeffs(θ, η, el::Inti.LagrangeElement, û)
	A = A_func(el, η)
	B = B_func(el, η)

	Jn = Jn_func(el, η)
	DJn_η = DJn(el, η)
	Dû_η = Dû(û, η)

	nx = Inti.normal(el, η)

	Λ = -1 / (4π)

	g₁ = A(θ) * (B(θ) ⋅ Jn(η) + A(θ) ⋅ (DJn_η * u_func(θ))) / norm(A(θ))^2

	b₀ = -Jn(η)
	b₁ = 3g₁ - DJn_η * u_func(θ)

	a₀ = b₀ * û(η)
	a₁ = b₁ * û(η) + b₀ * (Dû_η ⋅ u_func(θ))

	S₋₂ = -3 * (A(θ) ⋅ B(θ)) / norm(A(θ))^5
	S₋₃ = 1 / norm(A(θ))^3

	F₋₂ = (Λ * S₋₃ * a₀) ⋅ nx
	F₋₁ = Λ * (S₋₂ * a₀ + S₋₃ * a₁) ⋅ nx
	return F₋₁, F₋₂
end

"""
Compute only F₋₁ the old way (analytically with F₋₂) such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for Laplace hypersingular kernel.
"""
function LaplaceHypersingularClosedFormF₋₁(args...; kwargs...)
	f, _ = _laplace_hypersingular_closed_form_coeffs(args...; kwargs...)
	return f
end

"""
Compute only F₋₂ the old way (analytically with F₋₁) such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for Laplace hypersingular kernel.
"""
function LaplaceHypersingularClosedFormF₋₂(args...; kwargs...)
	_, f = _laplace_hypersingular_closed_form_coeffs(args...; kwargs...)
	return f
end

function ElastostaticAdjointDoubleLayerClosedFormCoeffs()
	notimplemented()
end

@memoize function _elastostatic_hypersingular_closed_form_coeffs(θ, η, el::Inti.LagrangeElement, û; λ, μ)
	ν = λ / (2 * (λ + μ))
	β = 3
	α = β - 1
	Λ = 1 / (4 * π * α * (1 - ν))

	A = A_func(el, η)
	B = B_func(el, η)

	Jn = Jn_func(el, η)
	DJn_η = DJn(el, η)
	Dû_η = Dû(û, η)

	nx = Inti.normal(el, η)

	S₋₂ = -3 * A(θ) ⋅ B(θ) / norm(A(θ))^5
	S₋₃ = 1 / norm(A(θ))^3

	function V_coeffs_12(i, k, j)
		δᵢⱼ = i == j
		δᵢₖ = i == k
		δⱼₖ = j == k
		d₀ = A(θ) / norm(A(θ))
		d₁ = B(θ) / norm(A(θ)) - A(θ) * (A(θ) ⋅ B(θ)) / norm(A(θ))^3
		J₀ = Jn(η)
		J₁ = DJn_η * u_func(θ)

		dᵢ₀ = d₀[i]
		dⱼ₀ = d₀[j]
		dₖ₀ = d₀[k]

		dᵢ₁ = d₁[i]
		dⱼ₁ = d₁[j]
		dₖ₁ = d₁[k]

		Jᵢ₀ = J₀[i]
		Jⱼ₀ = J₀[j]
		Jₖ₀ = J₀[k]

		Jᵢ₁ = J₁[i]
		Jⱼ₁ = J₁[j]
		Jₖ₁ = J₁[k]

		aᵢⱼ₀ = Jᵢ₀ * dⱼ₀ - Jⱼ₀ * dᵢ₀
		aᵢⱼ₁ = Jᵢ₁ * dⱼ₀ + Jᵢ₀ * dⱼ₁ - Jⱼ₁ * dᵢ₀ - Jⱼ₀ * dᵢ₁
		bᵢⱼₖ₀ = -Jᵢ₀ * δⱼₖ + Jⱼ₀ * δᵢₖ - Jₖ₀ * δᵢⱼ
		bᵢⱼₖ₁ = -Jᵢ₁ * δⱼₖ + Jⱼ₁ * δᵢₖ - Jₖ₁ * δᵢⱼ
		dᵢⱼₖ₀ = Jₖ₀ * dᵢ₀ * dⱼ₀
		dᵢⱼₖ₁ = Jₖ₁ * dᵢ₀ * dⱼ₀ + Jₖ₀ * dᵢ₁ * dⱼ₀ + Jₖ₀ * dᵢ₀ * dⱼ₁
		gᵢⱼₖ₀ = dᵢ₀ * dⱼ₀ * dₖ₀
		gᵢⱼₖ₁ = dᵢ₁ * dⱼ₀ * dₖ₀ + dᵢ₀ * dⱼ₀ * dₖ₁ + dᵢ₀ * dⱼ₁ * dₖ₀

		h₀ = J₀ ⋅ d₀
		h₁ = J₁ ⋅ d₀ + J₀ ⋅ d₁

		kᵢⱼₖ₀ = (1 - 2 * ν) * (β * dₖ₀ * aᵢⱼ₀ + bᵢⱼₖ₀) - β * dᵢⱼₖ₀ +
				β * h₀ * ((α + 3) * gᵢⱼₖ₀ + (1 - 2 * ν) * δᵢⱼ * dₖ₀ - δᵢₖ * dⱼ₀ - δⱼₖ * dᵢ₀)

		kᵢⱼₖ₁ = β * h₁ * ((α + 3) * gᵢⱼₖ₀ + (1 - 2 * ν) * δᵢⱼ * dₖ₀ - δᵢₖ * dⱼ₀ - δⱼₖ * dᵢ₀) +
				(1 - 2 * ν) * (β * dₖ₁ * aᵢⱼ₀ + β * aᵢⱼ₁ * dₖ₀ + bᵢⱼₖ₁) - β * dᵢⱼₖ₁ + β * h₀ * ((α + 3) * gᵢⱼₖ₁ + (1 - 2 * ν) * δᵢⱼ * dₖ₁ - δᵢₖ * dⱼ₁ - δⱼₖ * dᵢ₁)

		V₋₁ = -Λ * (S₋₂ * û(η) * kᵢⱼₖ₀ + S₋₃ * (Dû_η ⋅ u_func(θ) * kᵢⱼₖ₀ + û(η) * kᵢⱼₖ₁))
		V₋₂ = -Λ * S₋₃ * û(η) * kᵢⱼₖ₀
		return V₋₁, V₋₂
	end
	function H_coeff_12(i, j)
		h₋₁ = 0.0
		h₋₂ = 0.0
		for (b, k, ℓ) in Iterators.product(1:3, 1:3, 1:3)
			V₋₁, V₋₂ = V_coeffs_12(i, k, j)
			C_ibkℓ = hooke_tensor_iso(i, b, k, ℓ; λ, μ)
			h₋₁ += C_ibkℓ * nx[b] * V₋₁
			h₋₂ += C_ibkℓ * nx[b] * V₋₂
		end
		return h₋₁, h₋₂
	end
	F₋₁, F₋₂ = zeros(3, 3), zeros(3, 3)
	for (i, j) in Iterators.product(1:3, 1:3)
		h₋₁, h₋₂ = H_coeff_12(i, j)
		F₋₁[i, j] += h₋₁
		F₋₂[i, j] += h₋₂
	end
	return F₋₁, F₋₂
end

"""
Compute only F₋₁ the old way (analytically with F₋₂) such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for the elastostatic hypersingular kernel.
"""
function ElastostaticHypersingularClosedFormF₋₁(args...; kwargs...)
	f, _ = _elastostatic_hypersingular_closed_form_coeffs(args...; kwargs...)
	return f
end

"""
Compute only F₋₂ the old way (analytically with F₋₁) such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for the elastostatic hypersingular kernel.
"""
function ElastostaticHypersingularClosedFormF₋₂(args...; kwargs...)
	_, f = _elastostatic_hypersingular_closed_form_coeffs(args...; kwargs...)
	return f
end
