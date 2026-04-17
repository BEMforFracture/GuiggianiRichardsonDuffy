import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using ForwardDiff

# Configuration (MWE)
target_rel_error = 1e-9
max_points = 200

# Fixes demandes
n_rho = 4
n_a = 4
quad_rho = Inti.GaussLegendre(n_rho)
quad_a = Inti.GaussLegendre(n_a)

# Setup element (repris de duffy-polar-comparison.jl)
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)

# Point source fixe (MWE)
x̂ = SVector(0.1, 0.1)

# Densite et noyaux
û = ξ -> 1.0
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
SK = GRD.SplitKernel(K_base)
method = GRD.AutoDiffExpansion()

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

function duffy_guiggiani_singular_integral(SK, û, x̂, el, ori, quad_a, quad_b)
	x = el(x̂)
	Dτ = Inti.jacobian(el, x̂)
	nx = Inti._normal(Dτ, ori)
	D²τ = Inti.hessian(el, x̂)
	qx = (coords = x, normal = nx)
	N = length(x)

	duffy = 0.0
	for (ξᴵ, ξᴵᴵ, _) in duffy_decomposition(ref_domain)
		surface = (ξᴵ[1] - x̂[1]) * (ξᴵᴵ[2] - ξᴵ[2]) - (ξᴵᴵ[1] - ξᴵ[1]) * (ξᴵ[2] - x̂[2])

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
			return map(v -> K̂ * v, v) * surface * μ / norm(AB)^3
		end

		I_ab = quad_b() do (b,)
			c = b * (ξᴵᴵ - ξᴵ) + ξᴵ - x̂
			A = Dτ * c
			nA = norm(A)
			B = ntuple(i -> transpose(c) * D²τ[i, :, :] * c, N) |> SVector
			β = 1 / nA
			γ_over_β_squared = -A ⋅ B / nA^2

			func = t -> _func(t, b, c, A, B)
			f₋₂ = func(0.0)
			f₋₁ = ForwardDiff.derivative(func, 0.0)

			I_a = quad_a() do (a,)
				return (func(a) - f₋₂ - a * f₋₁) / a^2
			end

			return I_a - f₋₁ * log(abs(β)) - f₋₂ * (γ_over_β_squared + 1)
		end

		duffy += I_ab
	end

	return duffy
end

rel_error(I_num, I_ref) = norm(I_num - I_ref) / norm(I_ref)

function min_n_theta_for_target(K, û, x̂, el, ori, quad_rho, method, I_ref, ϵ; max_n = 200)
	for nθ in 1:max_n
		quad_θ = Inti.GaussLegendre(nθ)
		I_num = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_θ, method)
		err = rel_error(I_num, I_ref)
		if err <= ϵ
			return nθ, err
		end
	end
	return nothing, NaN
end

function min_n_b_for_target(SK, û, x̂, el, ori, quad_a, I_ref, ϵ; max_n = 200)
	for n_b in 1:max_n
		quad_b = Inti.GaussLegendre(n_b)
		I_num = duffy_guiggiani_singular_integral(SK, û, x̂, el, ori, quad_a, quad_b)
		err = rel_error(I_num, I_ref)
		if err <= ϵ
			return n_b, err
		end
	end
	return nothing, NaN
end

I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

nθ, errθ = min_n_theta_for_target(
	K_base,
	û,
	x̂,
	el,
	1,
	quad_rho,
	method,
	I_exp,
	target_rel_error;
	max_n = max_points,
)

n_b, err_b = min_n_b_for_target(
	SK,
	û,
	x̂,
	el,
	1,
	quad_a,
	I_exp,
	target_rel_error;
	max_n = max_points,
)

println("=== MWE Duffy vs Polar (n_rho et n_a fixes) ===")
println("target_rel_error = ", target_rel_error)
println("x̂ = ", x̂)
println("n_rho fixe = ", n_rho, " ; n_a fixe = ", n_a)

if isnothing(nθ)
	println("Polar: cible non atteinte jusqu'a n_theta = ", max_points)
else
	println("Polar: n_theta minimal = ", nθ, " ; erreur relative = ", errθ)
end

if isnothing(n_b)
	println("Duffy: cible non atteinte jusqu'a n_b = ", max_points)
else
	println("Duffy: n_b minimal = ", n_b, " ; erreur relative = ", err_b)
end
