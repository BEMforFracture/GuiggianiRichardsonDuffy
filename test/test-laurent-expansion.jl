using Test
using StaticArrays
using LinearAlgebra
using Inti
import GuiggianiRichardsonDuffy as GRD

# Test configuration
x̂ = SVector(0.4, 0.6)  # Source point in reference coordinates

# Richardson extrapolation parameters
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 10,
)

# Test with various density functions
functions = (ξ -> 1.0, ξ -> ξ[1], ξ -> ξ[2] * ξ[1], ξ -> ξ[1]^2 - ξ[2])

# Setup test element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

@testset "Laurent Coefficients - Accuracy comparison" begin
	# Test kernels
	kernels = [
		("Laplace", Inti.HyperSingularKernel(Inti.Laplace(dim = 3))),
		("Elastostatic", Inti.HyperSingularKernel(Inti.Elastostatic(μ = 1.0, λ = 1.0, dim = 3))),
	]

	# Expansion methods to test (excluding analytical, which is the reference)
	methods_to_test = [
		("AutoDiff", GRD.AutoDiffExpansion()),
		("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params)),
		("FullRichardson", GRD.FullRichardsonExpansion(rich_params)),
	]

	for (kernel_name, K) in kernels
		@testset "$kernel_name kernel" begin
			# Create SplitKernel for non-analytical methods
			SK = GRD.SplitKernel(K)

			for fun in functions
				@testset "Density: $(fun)" begin
					û = fun

					# Compute analytical reference
					ℒ_ana = GRD.laurents_coeffs(K, el, ori, û, x̂, GRD.AnalyticalExpansion())

					for (method_name, method) in methods_to_test
						@testset "$method_name" begin
							# Use SplitKernel for these methods
							ℒ_test = GRD.laurents_coeffs(SK, el, ori, û, x̂, method)

							# Sample angular points
							N_θ = 100
							θs = range(0, 2π; length = N_θ)

							# Compute values
							vals_ana = [ℒ_ana(θ) for θ in θs]
							vals_test = [ℒ_test(θ) for θ in θs]

							# Extract coefficients
							F₋₂_ana = [v[1] for v in vals_ana]
							F₋₁_ana = [v[2] for v in vals_ana]
							F₋₂_test = [v[1] for v in vals_test]
							F₋₁_test = [v[2] for v in vals_test]

							# Compute relative errors
							error_F₋₂ = norm(F₋₂_ana - F₋₂_test, 2) / norm(F₋₂_ana, 2)
							error_F₋₁ = norm(F₋₁_ana - F₋₁_test, 2) / norm(F₋₁_ana, 2)

							# Test accuracy thresholds
							@test error_F₋₂ < 1e-11
							@test error_F₋₁ < 1e-6
						end
					end
				end
			end
		end
	end
end
