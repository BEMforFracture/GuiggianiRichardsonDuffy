import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using ForwardDiff
using BenchmarkTools

# ------------------------------------------------------------
# Fast evaluation of the first Duffy integrand depending on (a, b)
# No quadrature: only the pointwise integrand used in G₁.
# ------------------------------------------------------------

# User inputs
a = 0.3
b = 0.4
theta = 0.7
rho = 0.2

# Fixed geometry / source point (edit if needed)
quad1D = Inti.GaussLegendre(2)
x̂ = quad1D.nodes[1][1]
x̂ = SVector(x̂, x̂)
ori = 1

E = 210e9
ν = 0.3
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2ν))

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

op = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)
K_base = Inti.HyperSingularKernel(op)
SK = GRD.SplitKernel(K_base)
K_polar = GRD.polar_kernel_fun(K_base, el, û, x̂, ori)
L = GRD.laurents_coeffs(SK, el, ori, û, x̂, GRD.AutoDiffExpansion())

x = el(x̂)
Dτ = Inti.jacobian(el, x̂)
nx = Inti._normal(Dτ, ori)
D²τ = Inti.hessian(el, x̂)
qx = (coords = x, normal = nx)
N = length(x)

function return_vertices(τ)
	if τ == 1
		return SVector(1.0, 0.0), SVector(1.0, 1.0)
	elseif τ == 2
		return SVector(1.0, 1.0), SVector(0.0, 1.0)
	elseif τ == 3
		return SVector(0.0, 1.0), SVector(0.0, 0.0)
	else
		return SVector(0.0, 0.0), SVector(1.0, 0.0)
	end
end

function duffy_decomposition(::Inti.ReferenceSquare)
	ξᴵ, ξᴵᴵ = return_vertices(1)
	_, ξᴵᴵᴵ = return_vertices(2)
	_, ξᴵⱽ = return_vertices(3)
	return (
		(ξᴵ, ξᴵᴵ, 1),
		(ξᴵᴵ, ξᴵᴵᴵ, 2),
		(ξᴵᴵᴵ, ξᴵⱽ, 3),
		(ξᴵⱽ, ξᴵ, 4),
	)
end

function local_geometry_data(ξᴵ, ξᴵᴵ, b)
	c = b * (ξᴵᴵ - ξᴵ) + ξᴵ - x̂
	A = Dτ * c
	nA = norm(A)
	B = ntuple(i -> transpose(c) * D²τ[i, :, :] * c, N) |> SVector
	β = 1 / nA
	γ_over_β_squared = -(A ⋅ B) / nA^2
	return c, A, B, β, γ_over_β_squared
end

function _func(a, c, A, B, surface)
	ŷ = x̂ + a * c
	jac_y = Inti.jacobian(el, ŷ)
	ny = Inti._normal(jac_y, ori)
	y = el(ŷ)
	qy = (coords = y, normal = ny)
	μy = Inti._integration_measure(jac_y)

	AB = A + a / 2 * B
	Â = AB / norm(AB)
	_, K̂ = SK(qx, qy, Â)
	v = û(ŷ)

	return K̂ * v * surface * μy / norm(AB)^3
end

function singular_coeffs(f)
	f₋₂ = f(0.0)
	f₋₁ = similar(f₋₂)
	for I in eachindex(f₋₂)
		f₋₁[I] = ForwardDiff.derivative(a -> f(a)[I], 0.0)
	end
	return f₋₂, f₋₁
end

# Crée une closure avec coefficients singuliers pré-calculés
function create_duffy_integrand(ξᴵ, ξᴵᴵ, b)
	surface = (ξᴵ[1] - x̂[1]) * (ξᴵᴵ[2] - ξᴵ[2]) - (ξᴵᴵ[1] - ξᴵ[1]) * (ξᴵ[2] - x̂[2])
	c, A, B, _, _ = local_geometry_data(ξᴵ, ξᴵᴵ, b)
	
	# Pré-calculer les coefficients singuliers une seule fois
	f = a_ -> _func(a_, c, A, B, surface)
	f₋₂, f₋₁ = singular_coeffs(f)
	
	# Retourner une closure légère
	return function (a)
		return 1 / a^2 * (f(a) - f₋₂ - a * f₋₁)
	end
end

function first_integrand_polar(theta, rho)
	rho <= eps(Float64) && error("rho doit etre > 0 pour evaluer l'integrande polaire regularise")
	θ = mod2pi(theta)
	f₋₂, f₋₁ = L(θ)
	return K_polar(rho, θ) - f₋₂ / rho^2 - f₋₁ / rho
end

# Choose the first sector by default.
decompo = duffy_decomposition(ref_domain)
ξᴵ, ξᴵᴵ, τ = decompo[1]

# Pré-créer la closure Duffy (comme L pour polaire)
duffy_integrand = create_duffy_integrand(ξᴵ, ξᴵᴵ, b)

value_duffy = duffy_integrand(a)
value_polar = first_integrand_polar(theta, rho)

bench_duffy = @benchmark $duffy_integrand($a)
bench_polar = @benchmark first_integrand_polar($theta, $rho)

println("Sector τ=$τ")
println("a = $a, b = $b")
println("theta = $theta, rho = $rho")
println("Duffy first integrand value =")
println(value_duffy)
println("Polar first integrand value =")
println(value_polar)

println("\nBenchmark Duffy:")
println(bench_duffy)
println("\nBenchmark Polar:")
println(bench_polar)
