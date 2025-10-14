# = Laurent's expansion of the geometry-dependent part of any kernel : kernel agnostic = #

u_func(θ) = SVector(cos(θ), sin(θ))

function A_func(τ::Inti.ReferenceInterpolant, η)::Function
	Dτ = Inti.jacobian(τ, η)
	A(θ) = Dτ * u_func(θ)
	return A
end

function B_func(τ::Inti.ReferenceInterpolant, η)::Function
	D²τ = Inti.hessian(τ, η)
	B(θ) = 0.5 * custom_contraction(D²τ, u_func(θ) * transpose(u_func(θ)))
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
		J = Inti.jacobian(τ, ξ)
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
	Dû = ForwardDiff.gradient(û, η)
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
