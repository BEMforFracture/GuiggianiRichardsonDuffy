import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Test

# Configuration
x̂ = SVector(0.1, 0.1)  # Source point in reference coordinates
n_a = 10
n_b = 40
quad_a = Inti.GaussLegendre(n_a)
quad_b = Inti.GaussLegendre(n_b)

n_rho = 10
n_theta = 15
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

ref_domain = Inti.reference_domain(el)

# Reference value
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
SK = GRD.SplitKernel(K_base)

s = Inti.singularity_order(SK)
S = s + 1
sorder = Val(S)

ori = 1

x = el(x̂)
Dτ = Inti.jacobian(el, x̂)
nx = Inti._normal(Dτ, ori)
D²τ = Inti.hessian(el, x̂)
qx = (coords = x, normal = nx)
N = length(x)

function return_vertices(τ)
	if τ == 1
		return SVector(1, 0), SVector(1, 1)
	elseif τ == 2
		return SVector(1, 1), SVector(0, 1)
	elseif τ == 3
		return SVector(0, 1), SVector(0, 0)
	else
		return SVector(0, 0), SVector(1, 0)
	end
end

function duffy_decomposition(::Inti.ReferenceSquare)
	ξᴵ, ξᴵᴵ = return_vertices(1)
	_, ξᴵᴵᴵ = return_vertices(2)
	_, ξᴵⱽ = return_vertices(3)
	return (ξᴵ, ξᴵᴵ, 1),
	(ξᴵᴵ, ξᴵᴵᴵ, 2),
	(ξᴵᴵᴵ, ξᴵⱽ, 3),
	(ξᴵⱽ, ξᴵ, 4)
end

# Initialize accumulator
# T = SMatrix{3, 3, Float64}
T = Float64
acc = zero(T)

# Integrate over each angular sector
for (ξᴵ, ξᴵᴵ, τ) in duffy_decomposition(ref_domain)
	surface = (ξᴵ[1] - x̂[1]) ⋅ (ξᴵᴵ[2] - ξᴵ[2]) - (ξᴵᴵ[1] - ξᴵ[1]) ⋅ (ξᴵ[2] - x̂[2])

	_func = (a, b, c, A, B) -> begin
		ŷ = x̂ + a * c
		jac_y = Inti.jacobian(el, ŷ)
		ny = Inti._normal(jac_y, ori)
		y = el(ŷ)
		qy = (coords = y, normal = ny)
		μ = Inti._integration_measure(jac_y)
		AB = A + a / 2 * B
		Â = AB / norm(AB)
		_, K̂ = SK(qx, qy, Â)
		v = û(ŷ)
		map(v -> K̂ * v, v) * surface * μ / norm(AB)^3
	end
	I_ab = quad_b() do (b,)
		c = b * (ξᴵᴵ - ξᴵ) + ξᴵ - x̂
		A = Dτ * c
		nA = norm(A)
		B = ntuple(i -> transpose(c) * D²τ[i, :, :] * c, N) |> SVector
		β = 1 / nA
		γ_over_β_squared = -A ⋅ B / nA^2

		coeffs_ = func -> begin
			F2 = func(0.0)
			F1 = ForwardDiff.derivative(func, 0.0)
			return (F1, F2)
		end

		func = t -> _func(t, b, c, A, B)

		f₋₁, f₋₂ = coeffs_(func)

		I_a = quad_a() do (a,)
			return 1 / a^2 * (func(a) - f₋₂ - a * f₋₁)
		end

		return I_a - f₋₁ * log(abs(β)) - f₋₂ * (γ_over_β_squared + 1)
	end
	global acc += I_ab
end

@info "Computed integral: $acc"
@info "Expected integral: $expected_I"
@info "Relative error: $(norm(acc - expected_I)/ norm(expected_I))"
