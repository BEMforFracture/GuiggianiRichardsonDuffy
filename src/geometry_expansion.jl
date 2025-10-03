# = Laurent's expansion of the geometry-dependent part of any kernel : kernel agnostic = #

function u_func()::Function
	return θ -> SVector(cos(θ), sin(θ))
end

function A_func(τ::Inti.ReferenceInterpolant, η)::Function
	Dτ = Inti.jacobian(τ, η)
	u = u_func()
	A(θ) = Dτ ⋅ u(θ)
	return A
end

function B_func(τ::Inti.ReferenceInterpolant, η)::Function
	D²τ = Inti.hessian(τ, η)
	u = u_func()
	B(θ) = 0.5 * custom_contraction(D²τ, u(θ) ⊗ u(θ))
	return B
end

function Â_func(τ::Inti.ReferenceInterpolant, η)::Function
	A = A_func(τ, η)
	Â(θ) = A(θ) / norm(A(θ))
	return Â
end

function B̂_func(τ::Inti.ReferenceInterpolant, η)::Function
	B = B_func(τ, η)
	A = A_func(τ, η)
	B̂(θ) = (B(θ) / norm(A(θ)) - A(θ) * (A(θ) ⋅ B(θ)) / norm(A(θ))^3)
	return B̂
end

function Jn_func(τ::Inti.ReferenceInterpolant, η)::Function
	Jn = ξ -> begin
		J = Inti.jacobian(τ, η)
		n = Inti._normal(J)
		μ = Inti._integration_measure(J)
		return μ * n
	end
	return Jn
end

function DJn(τ::Inti.ReferenceInterpolant, η)
	Jn = Jn_func(τ, η)
	DJn = Inti.jacobian(Jn, η)
	return DJn
end

function Dû(û::Function, η)
	Dû = Inti.jacobian(û, η)
	return Dû
end

function β_func(τ::Inti.ReferenceInterpolant, η)::Function
	A = A_func(τ, η)
	β(θ) = 1 / norm(A(θ))
	return β
end

function γ_func(τ::Inti.ReferenceInterpolant, η)::Function
	A = A_func(τ, η)
	B = B_func(τ, η)
	γ(θ) = -3 * (A(θ) ⋅ B(θ)) / norm(A(θ))^5
	return γ
end
