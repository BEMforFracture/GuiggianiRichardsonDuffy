#= Closed forme formulas for laurent's coefficients within Guiggiani algorithm =#

function LaplaceAdjointDoubleLayerClosedFormCoeffs()
	notimplemented()
end

@memoize function LaplaceHypersingularClosedFormCoeffs(θ, η, el::Inti.LagrangeElement, û)
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
function LaplaceHypersingularClosedFormF₋₁(θ, η, el::Inti.LagrangeElement, û)
	f, _ = LaplaceHypersingularClosedFormCoeffs(θ, η, el::Inti.LagrangeElement, û)
	return f
end

"""
Compute only F₋₂ the old way (analytically with F₋₁) such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for Laplace hypersingular kernel.
"""
function LaplaceHypersingularClosedFormF₋₂(θ, η, el::Inti.LagrangeElement, û)
	_, f = LaplaceHypersingularClosedFormCoeffs(θ, η, el::Inti.LagrangeElement, û)
	return f
end

function ElastostaticAdjointDoubleLayerClosedFormCoeffs()
	notimplemented()
end

@memoize function ElastostaticHypersingularClosedFormCoeffs()
	β = 3
	α = 2
	Λ = 1 / (4 * π * α * (1 - ν))

	SS₋₂ = S₋₂(θ, η, el)
	SS₋₃ = S₋₃(θ, η, el)

	JJᵢ₀ = Jₖ₀(η, el, i)
	JJᵢ₁ = Jₖ₁(θ, η, el, i)

	JJⱼ₀ = Jₖ₀(η, el, j)
	JJⱼ₁ = Jₖ₁(θ, η, el, j)

	JJₖ₀ = Jₖ₀(η, el, k)
	JJₖ₁ = Jₖ₁(θ, η, el, k)

	JJ₁₀ = Jₖ₀(η, el, 1)
	JJ₁₁ = Jₖ₁(θ, η, el, 1)

	JJ₂₀ = Jₖ₀(η, el, 2)
	JJ₂₁ = Jₖ₁(θ, η, el, 2)

	JJ₃₀ = Jₖ₀(η, el, 3)
	JJ₃₁ = Jₖ₁(θ, η, el, 3)

	NNᵖ₀ = Nᵖ₀(η, N_p)
	NNᵖ₁ = Nᵖ₁(θ, η, N_p)

	ddᵢ₀ = dₖ₀(θ, η, el, i)
	ddᵢ₁ = dₖ₁(θ, η, el, i)

	ddⱼ₀ = dₖ₀(θ, η, el, j)
	ddⱼ₁ = dₖ₁(θ, η, el, j)

	ddₖ₀ = dₖ₀(θ, η, el, k)
	ddₖ₁ = dₖ₁(θ, η, el, k)

	dd₁₀ = dₖ₀(θ, η, el, 1)
	dd₁₁ = dₖ₁(θ, η, el, 1)

	dd₂₀ = dₖ₀(θ, η, el, 2)
	dd₂₁ = dₖ₁(θ, η, el, 2)

	dd₃₀ = dₖ₀(θ, η, el, 3)
	dd₃₁ = dₖ₁(θ, η, el, 3)

	δᵢⱼ = i == j ? 1 : 0
	δᵢₖ = i == k ? 1 : 0
	δⱼₖ = j == k ? 1 : 0

	hh₀ = JJ₁₀ * dd₁₀ + JJ₂₀ * dd₂₀ + JJ₃₀ * dd₃₀
	hh₁ = JJ₁₁ * dd₁₀ + JJ₂₁ * dd₂₀ + JJ₃₁ * dd₃₀ + JJ₁₀ * dd₁₁ + JJ₂₀ * dd₂₁ + JJ₃₀ * dd₃₁

	ggᵢⱼₖ₀ = ddᵢ₀ * ddⱼ₀ * ddₖ₀
	ggᵢⱼₖ₁ = ddᵢ₁ * ddⱼ₀ * ddₖ₀ + ddᵢ₀ * ddⱼ₁ * ddₖ₀ + ddᵢ₀ * ddⱼ₀ * ddₖ₁

	ddᵢⱼₖ₀ = JJₖ₀ * ddᵢ₀ * ddⱼ₀
	ddᵢⱼₖ₁ = JJₖ₁ * ddᵢ₀ * ddⱼ₀ + JJₖ₀ * ddᵢ₁ * ddⱼ₀ + JJₖ₀ * ddᵢ₀ * ddⱼ₁

	bbᵢⱼₖ₀ = -JJᵢ₀ * δⱼₖ + JJⱼ₀ * δᵢₖ - JJₖ₀ * δᵢⱼ
	bbᵢⱼₖ₁ = -JJᵢ₁ * δⱼₖ + JJⱼ₁ * δᵢₖ - JJₖ₁ * δᵢⱼ

	aaᵢⱼ₀ = JJᵢ₀ * ddⱼ₀ - JJⱼ₀ * ddᵢ₀
	aaᵢⱼ₁ = JJᵢ₁ * ddⱼ₀ - JJⱼ₁ * ddᵢ₀ + JJᵢ₀ * ddⱼ₁ - JJⱼ₀ * ddᵢ₁

	kkᵢⱼₖ₀ = (1 - 2 * ν) * (β * ddₖ₀ * aaᵢⱼ₀ + bbᵢⱼₖ₀) - β * ddᵢⱼₖ₀ +
			 β * hh₀ * ((α + 3) * ggᵢⱼₖ₀ + (1 - 2 * ν) * δᵢⱼ * ddₖ₀ - δᵢₖ * ddⱼ₀ - δⱼₖ * ddᵢ₀)

	kkᵢⱼₖ₁ =
		β * hh₁ * ((α + 3) * ggᵢⱼₖ₀ + (1 - 2 * ν) * δᵢⱼ * ddₖ₀ - δᵢₖ * ddⱼ₀ - δⱼₖ * ddᵢ₀) + (1 - 2 * ν) * (β * ddₖ₁ * aaᵢⱼ₀ + β * aaᵢⱼ₁ * ddₖ₀ + bbᵢⱼₖ₁) - β * ddᵢⱼₖ₁ + β * hh₀ * ((α + 3) * ggᵢⱼₖ₁ + (1 - 2 * ν) * δᵢⱼ * ddₖ₁ - δᵢₖ * ddⱼ₁ - δⱼₖ * ddᵢ₁)

	F₋₂ = -Λ * SS₋₃ * NNᵖ₀ * kkᵢⱼₖ₀
	F₋₁ = -Λ * (SS₋₂ * NNᵖ₀ * kkᵢⱼₖ₀ + SS₋₃ * (NNᵖ₁ * kkᵢⱼₖ₀ + NNᵖ₀ * kkᵢⱼₖ₁))
	return F₋₁, F₋₂
end

function ElastostaticHypersingularClosedFormF₋₁()
	f, _ = ElastostaticHypersingularClosedFormCoeffs()
	return f
end

function ElastostaticHypersingularClosedFormF₋₂()
	f, _ = ElastostaticHypersingularClosedFormCoeffs()
	return f
end
