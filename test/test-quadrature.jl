using Test
using LinearAlgebra
using StaticArrays
using Inti
import GuiggianiRichardsonDuffy as GRD

# Test configuration
N = 10  # Number of test points

# Richardson extrapolation parameters
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 5,
)

# Quadrature parameters
n_rho = 10
n_theta = 30

# Test points in reference domain
test_points = [SVector(0.5, 0.5) for _ in 1:N]

# Setup test element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

# Constant density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
K = GRD.SplitKernel(K_base)

@testset "Polar coordinates - Integrate ρ_max(θ) over square element" begin
	# This test verifies that the polar decomposition and quadrature work correctly
	# by integrating ρ over the reference element, which should equal 1.0

	ref_domain = Inti.reference_domain(el)
	quad_rho = Inti.GaussLegendre(10)
	quad_theta = Inti.GaussLegendre(40)

	# Test multiple source points
	N_test = 10
	ξ_min = 1 / N_test
	ξ_max = 1 - 1 / N_test
	ξ_range = range(ξ_min, ξ_max; length = N_test)
	η_range = range(ξ_min, ξ_max; length = N_test)

	for ξ in ξ_range, η in η_range
		x̂ = SVector(ξ, η)

		# Integrate ρ in polar coordinates
		F(ρ, θ) = ρ
		acc = 0.0

		for (theta_min, theta_max, rho_func) in Inti.polar_decomposition(ref_domain, x̂)
			Δθ = theta_max - theta_min
			I_theta = quad_theta() do (theta_ref,)
				θ = theta_min + theta_ref * Δθ
				ρ_max = rho_func(θ)

				I_rho = quad_rho() do (rho_ref,)
					ρ = ρ_max * rho_ref
					return F(ρ, θ)
				end

				return I_rho * ρ_max
			end

			acc += I_theta * Δθ
		end

		@test abs(acc - 1.0) < 1e-6
	end
end

@testset "Laplace hypersingular - Singular integral accuracy" begin
	# Compute expected values using analytical closed form
	expected_values = [GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el) for x̂ in test_points]

	# Setup quadrature rules
	quad_rho = Inti.GaussLegendre(n_rho)
	quad_theta = Inti.GaussLegendre(n_theta)

	@testset "FullRichardson method" begin
		method = GRD.FullRichardsonExpansion(rich_params)

		results = [GRD.guiggiani_singular_integral(
			K, û, x̂, el, ori, quad_rho, quad_theta, method,
		) for x̂ in test_points]

		errors = [abs(res - expected) / abs(expected)
				  for (res, expected) in zip(results, expected_values)
				  if expected != 0.0]

		for error in errors
			@test error < 1e-7
		end
	end

	@testset "AutoDiff method" begin
		method = GRD.AutoDiffExpansion()

		results = [GRD.guiggiani_singular_integral(
			K, û, x̂, el, ori, quad_rho, quad_theta, method,
		) for x̂ in test_points]

		errors = [abs(res - expected) / abs(expected)
				  for (res, expected) in zip(results, expected_values)
				  if expected != 0.0]

		for error in errors
			@test error < 1e-11
		end
	end

	@testset "SemiRichardson method" begin
		method = GRD.SemiRichardsonExpansion(rich_params)

		results = [GRD.guiggiani_singular_integral(
			K, û, x̂, el, ori, quad_rho, quad_theta, method,
		) for x̂ in test_points]

		errors = [abs(res - expected) / abs(expected)
				  for (res, expected) in zip(results, expected_values)
				  if expected != 0.0]

		for error in errors
			@test error < 1e-7
		end
	end

	@testset "Analytical method" begin
		method = GRD.AnalyticalExpansion()

		# Note: Need to use base kernel (not SplitKernel) for analytical
		results = [GRD.guiggiani_singular_integral(
			K_base, û, x̂, el, ori, quad_rho, quad_theta, method,
		) for x̂ in test_points]

		errors = [abs(res - expected) / abs(expected)
				  for (res, expected) in zip(results, expected_values)
				  if expected != 0.0]

		for error in errors
			@test error < 1e-11
		end
	end
end
