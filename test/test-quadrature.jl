using LinearAlgebra
using StaticArrays
using Inti
import GuiggianiRichardsonDuffy as GRD

# INPUTS

N = 10  # Nombre de points testés

### Richardson extrapolation parameters
maxeval = 5
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

### quadrature parameters
n_rho = 10
n_theta = 30

# END INPUTS

# Génération de N points aléatoires dans [0,1]²
# test_points = [SVector(rand(), rand()) for _ in 1:N]
test_points = [SVector(0.5, 0.5) for _ in 1:N]

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
û = ξ -> 1.0

K = GRD.SplitLaplaceHypersingular

kernel_kwargs = NamedTuple()
richardson_kwargs = (maxeval = maxeval, rtol = rtol, atol = atol, contract = contract, first_contract = first_contract, breaktol = breaktol)

@testset "Integrate ρ_max(θ) over a square element" begin
	ref_domain = Inti.reference_domain(el)
	n_rho = 10
	n_theta = 40
	N = 10
	ξ_min = 1 / N
	ξ_max = 1 - 1 / N
	ξ_range = range(ξ_min, ξ_max, length = N)
	η_range = range(ξ_min, ξ_max, length = N)

	for ξ in ξ_range, η in η_range
		x̂ = SVector(ξ, η)
		F(ρ, θ) = ρ
		quad_rho = Inti.GaussLegendre(; order = n_rho)
		quad_theta = Inti.GaussLegendre(; order = n_theta)
		acc = zero(F(1.0, 0.0))
		for (theta_min, theta_max, rho_func) in Inti.polar_decomposition(ref_domain, x̂)
			Δθ = theta_max - theta_min
			I_theta = quad_theta() do (theta_ref,)
				θ = theta_min + theta_ref * Δθ
				ρ_max = rho_func(θ)
				I_rho = quad_rho() do (rho_ref,)
					ρ = ρ_max * rho_ref
					return F(ρ, θ)
				end
				I_rho *= ρ_max
			end
			I_theta *= Δθ
			acc += I_theta
		end
		@test abs(acc - 1.0) < 1e-6
	end
end

@testset "Laplace hypersingular quadrature over $N points" begin
	# Calcul vectorisé des valeurs attendues
	expected_values = [GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el) for x̂ in test_points]
	@testset "Richardson" begin
		# Calcul vectorisé des résultats
		results_rich = [GRD.guiggiani_singular_integral(
			K,
			û,
			x̂,
			el,
			n_rho,
			n_theta;
			sorder = Val(-3),
			expansion = :full_richardson,
			kernel_kwargs = kernel_kwargs,
			richardson_kwargs = richardson_kwargs,
		) for x̂ in test_points]

		# Calcul vectorisé des erreurs relatives
		errors_rich = [abs(res - expected) / abs(expected) for (res, expected) in zip(results_rich, expected_values) if expected != 0.0]

		# Test que toutes les erreurs sont inférieures au seuil
		for (i, error) in enumerate(errors_rich)
			@test error < 1e-7
		end
	end

	@testset "Automatic Differentiation" begin
		# Calcul vectorisé des résultats
		results_auto = [GRD.guiggiani_singular_integral(
			K,
			û,
			x̂,
			el,
			n_rho,
			n_theta;
			sorder = Val(-3),
			expansion = :auto_diff,
			kernel_kwargs = kernel_kwargs,
		) for x̂ in test_points]

		# Calcul vectorisé des erreurs relatives
		errors_auto = [abs(res - expected) / abs(expected) for (res, expected) in zip(results_auto, expected_values) if expected != 0.0]

		# Test que toutes les erreurs sont inférieures au seuil
		for (i, error) in enumerate(errors_auto)
			@test error < 1e-11
		end
	end

	@testset "Semi-Richardson" begin

		results_semi = [GRD.guiggiani_singular_integral(
			K,
			û,
			x̂,
			el,
			n_rho,
			n_theta;
			sorder = Val(-3),
			expansion = :semi_richardson,
			kernel_kwargs = kernel_kwargs,
			richardson_kwargs = richardson_kwargs,
		) for x̂ in test_points]

		# Calcul vectorisé des erreurs relatives
		errors_semi = [abs(res - expected) / abs(expected) for (res, expected) in zip(results_semi, expected_values) if expected != 0.0]
		# Test que toutes les erreurs sont inférieures au seuil
		for (i, error) in enumerate(errors_semi)
			@test error < 1e-7
		end
	end

	@testset "Analytical" begin

		results_analytical = [GRD.guiggiani_singular_integral(
			K,
			û,
			x̂,
			el,
			n_rho,
			n_theta;
			sorder = Val(-3),
			expansion = :analytical,
			kernel_kwargs = kernel_kwargs,
		) for x̂ in test_points]

		# Calcul vectorisé des erreurs relatives
		errors_analytical = [abs(res - expected) / abs(expected) for (res, expected) in zip(results_analytical, expected_values) if expected != 0.0]
		# Test que toutes les erreurs sont inférieures au seuil
		for (i, error) in enumerate(errors_analytical)
			@test error < 1e-11
		end
	end
end
