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
	notimplemented()
end

function ElastostaticHypersingularClosedFormF₋₁()
	notimplemented()
end

function ElastostaticHypersingularClosedFormF₋₂()
	notimplemented()
end
