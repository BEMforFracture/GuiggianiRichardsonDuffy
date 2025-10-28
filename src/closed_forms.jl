#= Closed forme formulas for laurent's coefficients within Guiggiani algorithm =#

function LaurentCoeffsClosedForms(K::Inti.AbstractKernel, θ::Float64, η::SVector{D, T}, el::Inti.ReferenceInterpolant{D, SVector{N, T}}, û) where {T <: Real, N, D}
	return _laurents_coeffs_closed_forms(K, θ, η, el, û)
end

function _laurents_coeffs_closed_forms(K::Inti.SingleLayerKernel{T, <:Inti.Laplace{N}}, θ, η, el, û) where {T, N}
	error("Closed form Laurent coefficients not implemented for $(typeof(K)). Available for: $(ANALYTICAL_EXPANSIONS)")
end

function _laurents_coeffs_closed_forms(K::Inti.DoubleLayerKernel{T, <:Inti.Laplace{N}}, θ, η, el, û) where {T, N}
	error("Closed form Laurent coefficients not implemented for $(typeof(K)). Available for: $(ANALYTICAL_EXPANSIONS)")
end

function _laurents_coeffs_closed_forms(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Laplace{N}}, θ, η, el, û) where {T, N}
	error("Closed form Laurent coefficients not implemented for $(typeof(K)). Available for: $(ANALYTICAL_EXPANSIONS)")
end

@memoize function _laurents_coeffs_closed_forms(K::Inti.HyperSingularKernel{T, <:Inti.Laplace{N}}, θ, η, el::Inti.LagrangeElement, û) where {T, N}
	A = A_func(el, η)
	B = B_func(el, η)

	Jn = Jn_func(el, η)
	DJn_η = DJn(el, η)
	Dû_η = Dû(û, η)

	nx = Inti.normal(el, η)

	Λ = -1 / (4π)

	g₁ = A(θ) * (transpose(B(θ)) * Jn(η) + transpose(A(θ)) * (DJn_η * u_func(θ))) / norm(A(θ))^2

	b₀ = -Jn(η)
	b₁ = 3g₁ - DJn_η * u_func(θ)

	a₀ = b₀ * û(η)
	a₁ = b₁ * û(η) + b₀ * (transpose(Dû_η) * u_func(θ))

	S₋₂ = -3 * (transpose(A(θ)) * B(θ)) / norm(A(θ))^5
	S₋₃ = 1 / norm(A(θ))^3

	F₋₂ = Λ * S₋₃ * transpose(a₀) * nx
	F₋₁ = Λ * transpose(S₋₂ * a₀ + S₋₃ * a₁) * nx
	return F₋₂, F₋₁
end

function _laurents_coeffs_closed_forms(K::Inti.SingleLayerKernel{T, <:Inti.Elastostatic{N}}, θ, η, el, û) where {T, N}
	error("Closed form Laurent coefficients not implemented for $(typeof(K)). Available for: $(ANALYTICAL_EXPANSIONS)")
end

function _laurents_coeffs_closed_forms(K::Inti.DoubleLayerKernel{T, <:Inti.Elastostatic{N}}, θ, η, el, û) where {T, N}
	error("Closed form Laurent coefficients not implemented for $(typeof(K)). Available for: $(ANALYTICAL_EXPANSIONS)")
end

function _laurents_coeffs_closed_forms(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Elastostatic{N}}, θ, η, el, û) where {T, N}
	error("Closed form Laurent coefficients not implemented for $(typeof(K)). Available for: $(ANALYTICAL_EXPANSIONS)")
end

@memoize function _laurents_coeffs_closed_forms(K::Inti.HyperSingularKernel{T, <:Inti.Elastostatic{N}}, θ, η, el::Inti.LagrangeElement, û) where {T, N}
	λ = K.op.λ
	μ = K.op.μ
	ν = λ / (2 * (λ + μ))
	β = 3
	α = β - 1
	Λ = 1 / (4 * π * α * (1 - ν))

	A = A_func(el, η)
	B = B_func(el, η)

	Jn = Jn_func(el, η)
	DJn_η = DJn(el, η)
	Dû_η = Dû(û, η)

	nx = Inti.normal(el, η)

	S₋₂ = -3 * dot(A(θ), B(θ)) / norm(A(θ))^5
	S₋₃ = 1 / norm(A(θ))^3

	d₀ = A(θ) / norm(A(θ))
	d₁ = B(θ) / norm(A(θ)) - A(θ) * (dot(A(θ), B(θ)) / norm(A(θ))^3)
	J₀ = Jn(η)
	J₁ = DJn_η * u_func(θ)

	function V_coeffs_12(i, k, j)
		δᵢⱼ = i == j
		δᵢₖ = i == k
		δⱼₖ = j == k

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

		h₀ = dot(J₀, d₀)
		h₁ = dot(J₁, d₀) + dot(J₀, d₁)

		kᵢⱼₖ₀ = (1 - 2 * ν) * (β * dₖ₀ * aᵢⱼ₀ + bᵢⱼₖ₀) - β * dᵢⱼₖ₀ +
				β * h₀ * ((α + 3) * gᵢⱼₖ₀ + (1 - 2 * ν) * δᵢⱼ * dₖ₀ - δᵢₖ * dⱼ₀ - δⱼₖ * dᵢ₀)

		kᵢⱼₖ₁ = β * h₁ * ((α + 3) * gᵢⱼₖ₀ + (1 - 2 * ν) * δᵢⱼ * dₖ₀ - δᵢₖ * dⱼ₀ - δⱼₖ * dᵢ₀) +
				(1 - 2 * ν) * (β * dₖ₁ * aᵢⱼ₀ + β * aᵢⱼ₁ * dₖ₀ + bᵢⱼₖ₁) - β * dᵢⱼₖ₁ + β * h₀ * ((α + 3) * gᵢⱼₖ₁ + (1 - 2 * ν) * δᵢⱼ * dₖ₁ - δᵢₖ * dⱼ₁ - δⱼₖ * dᵢ₁)

		V₋₁ = -Λ * (S₋₂ * û(η) * kᵢⱼₖ₀ + S₋₃ * (Dû_η ⋅ u_func(θ) * kᵢⱼₖ₀ + û(η) * kᵢⱼₖ₁))
		V₋₂ = -Λ * S₋₃ * û(η) * kᵢⱼₖ₀
		return V₋₁, V₋₂
	end
	function H_coeff_12(i, j)
		h₋₁ = 0.0
		h₋₂ = 0.0
		for (b, k, ℓ) in Iterators.product(1:3, 1:3, 1:3)
			V₋₁, V₋₂ = V_coeffs_12(k, ℓ, j)
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
	return F₋₂, F₋₁
end

function hypersingular_laplace_integral_on_plane_element(x, el)
	!(is_plane(el)) && throw(ArgumentError("Element is not planar."))
	res = 0.0
	_Y = el.vals
	n = length(_Y)
	# renumbering to use cyclic indexing
	if n == 4
		_Y = [_Y[1], _Y[2], _Y[4], _Y[3]]
	end
	Y = SVector(_Y..., _Y[1])  # close the polygon
	u = [y - x for y in Y]
	for p in 1:n
		res += (norm(u[p]) + norm(u[p+1])) * norm(cross(u[p], u[p+1])) / (norm(u[p]) * norm(u[p+1]) * (norm(u[p]) * norm(u[p+1]) + dot(u[p], u[p+1])))
	end
	return -res / (4π)
end
