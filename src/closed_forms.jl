#= Closed forme formulas for laurent's coefficients within Guiggiani algorithm =#

function LaplaceAdjointDoubleLayerClosedFormCoeffs()
end

@memoize function LaplaceHypersingularClosedFormCoeffs(θ, η, el::Inti.LagrangeElement, û)
	A = A_func(el, η)
	B = B_func(el, η)

	Jn = Jn_func(el, η)
	DJn_η = DJn(el, η)
	Dû_η = Dû(û, η)

	Λ = -1 / (4π)

	g₁ = A(θ) * (B(θ) ⋅ Jn(η) + A(θ) ⋅ (DJn_η * u_func(θ)))

	b₀ = -Jn(η) * û(η)
	b₁ = 3g₁ - DJn_η * u_func(θ)

	a₀ = -Jn(η) * û(η)
	a₁ = b₁ * û(η) + b₀ * (Dû_η ⋅ u_func(θ))

	S₋₂ = -3 * (A(θ) ⋅ B(θ)) / norm(A(θ))^5
	S₋₃ = 1 / norm(A(θ))^3

	F₋₂ = Λ * S₋₃ * a₀
	F₋₁ = Λ * (S₋₂ * a₀ + S₋₃ * a₁)
	return F₋₁, F₋₂
end

"""
Compute only F₋₁ such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for Laplace hypersingular kernel.
"""
function LaplaceHypersingularClosedFormF₋₁(θ, η, el::Inti.LagrangeElement, û)
	f, _ = LaplaceHypersingularClosedFormCoeffs(θ, η, el::Inti.LagrangeElement, û)
	return f
end

"""
Compute only F₋₂ such that F(ρ, θ) := ρ × J × Nᵖ × Vᵢₖⱼ = F₋₁ / ρ + F₋₂ / ρ² + O(1), for Laplace hypersingular kernel.
"""
function LaplaceHypersingularClosedFormF₋₂(θ, η, el::Inti.LagrangeElement, û)
	_, f = LaplaceHypersingularClosedFormCoeffs(θ, η, el::Inti.LagrangeElement, û)
	return f
end

function ElastostaticHypersingularClosedFormCoeffs()
	notimplemented()
end

function ElastostaticAdjointDoubleLayerClosedFormCoeffs()
	notimplemented()
end
