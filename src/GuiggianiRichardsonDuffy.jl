module GuiggianiRichardsonDuffy

using LinearAlgebra
using StaticArrays
using Inti
using Richardson

include("utils.jl")
include("kernels.jl")

@info "Loading GuiggianiRichardsonDuffy.jl"

"""
	polar_kernel_fun(K::Inti.AbstractKernel, el::Inti.ReferenceInterpolant, û, x̂)

	Given a kernel `K`, a reference element `el`, a function `û` defined on the reference element, and a point `x̂` on the reference element, returns a function `F` that computes the complete kernel in polar coordinates centered at `x̂` : F(ρ, θ) = K(x, y) * J(ŷ) * ρ * û(ŷ) where `x = el(x̂)`, `ŷ = x̂ + ρ * (cos(θ), sin(θ))`, `y = el(ŷ)`, and `J(ŷ)` is the integration measure at `ŷ`. `F` will be called as `F(ρ, θ)`. `K` has to be called as `K(qx, qy)` where `qx = (coords = x, normal = nx)` and `qy = (coords = y, normal = ny)` are cartesian points with their normals.
"""
function polar_kernel_fun(K, el::Inti.ReferenceInterpolant, û, x̂)
	x = el(x̂)
	ori = 1
	jac_x = Inti.jacobian(el, x̂)
	nx = Inti._normal(jac_x, ori)
	qx = (coords = x, normal = nx)
	function F(ρ, θ)
		s, c = sincos(θ)
		ŷ = x̂ + ρ * SVector(c, s)
		y = el(ŷ)
		jac_y = Inti.jacobian(el, ŷ)
		ny = Inti._normal(jac_y, ori)
		μ = Inti._integration_measure(jac_y)
		qy = (coords = y, normal = ny)
		return K(qx, qy) * μ * ρ * û(ŷ)
	end
	return F
end

"""
	A_factor_0_fun(el::Inti.ReferenceInterpolant, x̂)

	Given a reference element `el` and a point `x̂` on the reference element, returns the function `A(θ)` such that the relative position vector `y - x` can be expressed as `y - x = ρ * A(θ) + O(ρ^2)` where `x = el(x̂)`, `ŷ = x̂ + ρ * (cos(θ), sin(θ))`, and `y = el(ŷ)`. `A` will be called as `A(θ)`.
"""
function A_factor_0_fun(el::Inti.ReferenceInterpolant, x̂)
	Dτ = Inti.jacobian(el, x̂)
	u(θ) = SVector(cos(θ), sin(θ))
	A(θ) = Dτ ⋅ u(θ)
	return A
end

"""
	f_minus_two(K̂::Inti.HyperSingularKernel, el::Inti.ReferenceInterpolant, û, x̂)

	Computes the laurent coefficient f_{-2} for the kernel K = K̂ / r³ in polar coordinates centered at x̂, where K̂ is a smooth kernel, el is a reference element, û is a function defined on the reference element, and x̂ is a point on the reference element.

	K̂ as to be called as K̂(r̂, qx, qy) where r̂ is the normalized relative position vector, qx = (coords = x, normal = nx) and qy = (coords = y, normal = ny).
"""
function f_minus_two_fun(K̂, el::Inti.ReferenceInterpolant, û, x̂)
	x = el(x̂)
	jac_x = Inti.jacobian(el, x̂)
	ori = 1
	nx = Inti._normal(jac_x, ori)
	qx = (coords = x, normal = nx)
	μ = Inti._integration_measure(jac_x)
	A = A_factor_0_fun(el, x̂)
	function f_minus_two(θ)
		Aθ = A(θ)
		Âθ = Aθ / norm(Aθ)
		return K̂(Âθ, qx, qx) * μ * û(x̂) / norm(Aθ)^3
	end
	return f_minus_two
end

"""
	rho_fun(ref_domain::Inti.ReferenceDomain, η)

	Given a reference domain `ref_domain` and a point `η` in the reference domain, returns the function `ρ(θ)` that gives the distance from `η` to the boundary of the reference domain in the direction `θ`. `ρ` will be called as `ρ(θ)`.
"""
function rho_fun(ref_domain, η)
	decompo = Inti.polar_decomposition(ref_domain, η)
	function ρ(θ)
		if decompo[1][1] ≤ θ < decompo[1][2]
			return decompo[1][3](θ)
		elseif decompo[2][1] ≤ θ < decompo[2][2]
			return decompo[2][3](θ)
		elseif decompo[3][1] ≤ θ < decompo[3][2]
			return decompo[3][3](θ)
		else
			return decompo[4][3](θ)
		end
	end
	return ρ
end

"""
	function f_minus_one_fun(fun, rho_max_fun, f₋₂; first_contract, contract)

	Given a function `fun(ρ)`, a function `rho_max_fun(θ)` that gives the maximum value of `ρ` for each `θ`, and the laurent coefficient `f₋₂`, returns the function `f₋₁(θ)` that computes the laurent coefficient `f₋₁` using Richardson extrapolation. The parameters `first_contract` and `contract` control the extrapolation process.
"""
function f_minus_one_fun(fun, rho_max_fun, f₋₂; first_contract, contract)
	function f_minus_one(θ)
		h = rho_max_fun(θ) * first_contract
		f₋₁, e₋₁ = extrapolate(h; x0 = 0, contract = contract) do x
			return x * fun(x) - f₋₂ / x
		end
	end
	return f_minus_one
end

end # module GuiggianiRichardsonDuffy
