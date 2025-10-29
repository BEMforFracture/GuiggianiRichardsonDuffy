using Test
import GuiggianiRichardsonDuffy as GRD
using Inti
using StaticArrays

@testset "Smoke tests - SplitKernel construction" begin
	@testset "Laplace kernels" begin
		op = Inti.Laplace(; dim = 3)
		K_SL = Inti.SingleLayerKernel(op)
		K_DL = Inti.DoubleLayerKernel(op)
		K_ADL = Inti.AdjointDoubleLayerKernel(op)
		K_HS = Inti.HyperSingularKernel(op)

		# Test de wrapping
		@test_nowarn GRD.SplitKernel(K_SL)
		@test_nowarn GRD.SplitKernel(K_DL)
		@test_nowarn GRD.SplitKernel(K_ADL)
		@test_nowarn GRD.SplitKernel(K_HS)

		SK_SL = GRD.SplitKernel(K_SL)
		SK_DL = GRD.SplitKernel(K_DL)
		SK_ADL = GRD.SplitKernel(K_ADL)
		SK_HS = GRD.SplitKernel(K_HS)

		# Test que singularity_order fonctionne
		@test Inti.singularity_order(SK_SL) == -1
		@test Inti.singularity_order(SK_DL) == -2
		@test Inti.singularity_order(SK_ADL) == -2
		@test Inti.singularity_order(SK_HS) == -3
	end
	@testset "Elastostatic kernels" begin
		op = Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)
		K_SL = Inti.SingleLayerKernel(op)
		K_DL = Inti.DoubleLayerKernel(op)
		K_ADL = Inti.AdjointDoubleLayerKernel(op)
		K_HS = Inti.HyperSingularKernel(op)

		# Test de wrapping
		@test_nowarn GRD.SplitKernel(K_SL)
		@test_nowarn GRD.SplitKernel(K_DL)
		@test_nowarn GRD.SplitKernel(K_ADL)
		@test_nowarn GRD.SplitKernel(K_HS)

		SK_SL = GRD.SplitKernel(K_SL)
		SK_DL = GRD.SplitKernel(K_DL)
		SK_ADL = GRD.SplitKernel(K_ADL)
		SK_HS = GRD.SplitKernel(K_HS)

		# Test que singularity_order fonctionne
		@test Inti.singularity_order(SK_SL) == -1
		@test Inti.singularity_order(SK_DL) == -2
		@test Inti.singularity_order(SK_ADL) == -2
		@test Inti.singularity_order(SK_HS) == -3
	end
end

@testset "Smoke tests - SplitKernels evaluation" begin
	@testset "Laplace" begin
		@testset "Basic evaluation (non-singular)" begin
			K = Inti.HyperSingularKernel(Inti.Laplace(; dim = 3))
			SK = GRD.SplitKernel(K)

			# Points de test
			qx = (coords = SVector(0.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))
			qy = (coords = SVector(1.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))

			# Test que l'évaluation fonctionne
			@test_nowarn SK(qx, qy)

			# Test que le résultat est un tuple
			result = SK(qx, qy)
			@test result isa Tuple{<:Real, <:Any}

			# Test que prod fonctionne
			@test_nowarn prod(result)
		end
		@testset "Singular point evaluation" begin
			K = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
			SK = GRD.SplitKernel(K)

			qx = (coords = SVector(0.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))
			r̂ = SVector(1.0, 0.0, 0.0)

			# Test avec direction fournie
			@test_nowarn SK(qx, qx, r̂)
		end
	end
	@testset "Elastostatic" begin
		@testset "Basic evaluation (non-singular)" begin
			op = Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)
			K = Inti.HyperSingularKernel(op)
			SK = GRD.SplitKernel(K)

			# Points de test
			qx = (coords = SVector(0.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))
			qy = (coords = SVector(1.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))

			# Test que l'évaluation fonctionne
			@test_nowarn SK(qx, qy)

			# Test que le résultat est un tuple
			result = SK(qx, qy)
			@test result isa Tuple{<:Real, <:SMatrix{3, 3, Float64, 9}}

			# Test que prod fonctionne
			@test_nowarn prod(result)
		end
		@testset "Singular point evaluation" begin
			op = Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)
			K = Inti.HyperSingularKernel(op)
			SK = GRD.SplitKernel(K)

			qx = (coords = SVector(0.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))
			r̂ = SVector(1.0, 0.0, 0.0)

			# Test avec direction fournie
			@test_nowarn SK(qx, qx, r̂)
		end
	end

	@testset "Smoke tests - Expansion types construction" begin
		@test_nowarn GRD.RichardsonParams()
		@test_nowarn GRD.AnalyticalExpansion()
		@test_nowarn GRD.AutoDiffExpansion()

		params = GRD.RichardsonParams()
		@test_nowarn GRD.FullRichardsonExpansion(params)
		@test_nowarn GRD.SemiRichardsonExpansion(params)

		# Vérifier que les paramètres sont bien stockés
		full_method = GRD.FullRichardsonExpansion(params)
		@test full_method.richardson_params === params
	end

	@testset "Smoke tests - compute_coefficients" begin
		K = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
		δ = 0.5
		z = 0.0
		y¹ = SVector(-1.0, -1.0, z)
		y² = SVector(1.0 + δ, -1.0, z)
		y³ = SVector(-1.0, 1.0, z)
		y⁴ = SVector(1.0 - δ, 1.0, z)
		nodes = (y¹, y², y³, y⁴)

		el = Inti.LagrangeSquare(nodes)
		x̂ = SVector(0.5, 0.5)
		û = η -> 1.0
		θ = π / 4

		@testset "AnalyticalExpansion" begin
			method = GRD.AnalyticalExpansion()

			ℒ = @test_nowarn GRD._create_laurent_coeffs_function(method, K, el, û, x̂)
			f₋₂ = ℒ(θ)[1]
			f₋₁ = ℒ(θ)[2]
			@test f₋₂ isa Real
			@test f₋₁ isa Real
		end

		@testset "AutoDiffExpansion" begin
			method = GRD.AutoDiffExpansion()
			SK = GRD.SplitKernel(K)

			ℒ = @test_nowarn GRD._create_laurent_coeffs_function(method, K, el, û, x̂)
			f₋₂ = ℒ(θ)[1]
			f₋₁ = ℒ(θ)[2]
			@test f₋₂ isa Real
			@test f₋₁ isa Real
		end

		@testset "FullRichardsonExpansion" begin
			method = GRD.FullRichardsonExpansion(GRD.RichardsonParams())
			SK = GRD.SplitKernel(K)

			ℒ = @test_nowarn GRD._create_laurent_coeffs_function(method, K, el, û, x̂)
			f₋₂ = ℒ(θ)[1]
			f₋₁ = ℒ(θ)[2]
			@test f₋₂ isa Real
			@test f₋₁ isa Real
		end

		@testset "SemiRichardsonExpansion" begin
			method = GRD.SemiRichardsonExpansion(GRD.RichardsonParams())
			SK = GRD.SplitKernel(K)

			ℒ = @test_nowarn GRD._create_laurent_coeffs_function(method, K, el, û, x̂)
			f₋₂ = ℒ(θ)[1]
			f₋₁ = ℒ(θ)[2]
			@test f₋₂ isa Real
			@test f₋₁ isa Real
		end
	end

	@testset "Smoke tests - Closed forms" begin
		K_lap = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
		K_elast = Inti.HyperSingularKernel(Inti.Elastostatic(μ = 1.0, λ = 1.0, dim = 3))

		δ = 0.5
		z = 0.0
		y¹ = SVector(-1.0, -1.0, z)
		y² = SVector(1.0 + δ, -1.0, z)
		y³ = SVector(-1.0, 1.0, z)
		y⁴ = SVector(1.0 - δ, 1.0, z)
		nodes = (y¹, y², y³, y⁴)

		el = Inti.LagrangeSquare(nodes)
		x̂ = SVector(0.5, 0.5)
		û = η -> 1.0
		θ = π / 4

		@testset "Laplace hypersingular" begin
			@test_nowarn GRD._laurents_coeffs_closed_forms(K_lap, θ, x̂, el, û)
			f₋₂, f₋₁ = GRD._laurents_coeffs_closed_forms(K_lap, θ, x̂, el, û)
			@test f₋₂ isa Real
			@test f₋₁ isa Real
		end

		@testset "Elastostatic hypersingular" begin
			@test_nowarn GRD._laurents_coeffs_closed_forms(K_elast, θ, x̂, el, û)
			f₋₂, f₋₁ = GRD._laurents_coeffs_closed_forms(K_elast, θ, x̂, el, û)
			@test f₋₂ isa AbstractMatrix
			@test f₋₁ isa AbstractMatrix
		end

		@testset "Not implemented kernels" begin
			K_SL = Inti.SingleLayerKernel(Inti.Laplace(dim = 3))
			@test_throws ErrorException GRD._laurents_coeffs_closed_forms(K_SL, θ, x̂, el, û)
		end
	end

	@testset "Smoke tests - API functions" begin
		# Setup commun
		K = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
		δ = 0.5
		z = 0.0
		y¹ = SVector(-1.0, -1.0, z)
		y² = SVector(1.0 + δ, -1.0, z)
		y³ = SVector(-1.0, 1.0, z)
		y⁴ = SVector(1.0 - δ, 1.0, z)
		nodes = (y¹, y², y³, y⁴)
		el = Inti.LagrangeSquare(nodes)
		x̂ = SVector(0.5, 0.5)
		û = η -> 1.0

		@testset "polar_kernel_fun" begin
			# Test avec kernel normal
			@test_nowarn GRD.polar_kernel_fun(K, el, û, x̂)
			K_polar = GRD.polar_kernel_fun(K, el, û, x̂)

			# Test que c'est une fonction
			@test K_polar isa Function

			# Test évaluation
			@test_nowarn K_polar(0.1, π / 4)
			result = K_polar(0.1, π / 4)
			@test result isa Real

			# Test avec SplitKernel
			SK = GRD.SplitKernel(K)
			Kprod = (qx, qy) -> prod(SK(qx, qy))
			@test_nowarn GRD.polar_kernel_fun(Kprod, el, û, x̂)
		end

		@testset "rho_fun" begin
			ref_domain = Inti.reference_domain(el)
			@test_nowarn GRD.rho_fun(ref_domain, x̂)

			ρ = GRD.rho_fun(ref_domain, x̂)
			@test ρ isa Function

			# Test évaluation
			@test_nowarn ρ(π / 4)
			result = ρ(π / 4)
			@test result isa Real
			@test result > 0  # Distance doit être positive
		end

		@testset "laurents_coeffs - default constructor" begin
			# Test avec constructeurs par défaut
			@test_nowarn GRD.FullRichardsonExpansion()
			@test_nowarn GRD.SemiRichardsonExpansion()

			method_full = GRD.FullRichardsonExpansion()
			method_semi = GRD.SemiRichardsonExpansion()

			@test method_full isa GRD.FullRichardsonExpansion
			@test method_semi isa GRD.SemiRichardsonExpansion
		end

		@testset "laurents_coeffs - different methods" begin
			# FullRichardson (par défaut)
			@test_nowarn GRD.laurents_coeffs(K, el, û, x̂)
			ℒ = GRD.laurents_coeffs(K, el, û, x̂)
			@test ℒ isa Function
			f₋₂, f₋₁ = ℒ(π / 4)
			@test f₋₂ isa Real
			@test f₋₁ isa Real

			# Analytical
			@test_nowarn GRD.laurents_coeffs(K, el, û, x̂, GRD.AnalyticalExpansion())
			ℒ_anal = GRD.laurents_coeffs(K, el, û, x̂, GRD.AnalyticalExpansion())
			f₋₂, f₋₁ = ℒ_anal(π / 4)
			@test f₋₂ isa Real
			@test f₋₁ isa Real

			# AutoDiff (nécessite SplitKernel)
			@test_nowarn GRD.laurents_coeffs(K, el, û, x̂, GRD.AutoDiffExpansion())
			ℒ_ad = GRD.laurents_coeffs(K, el, û, x̂, GRD.AutoDiffExpansion())
			f₋₂, f₋₁ = ℒ_ad(π / 4)
			@test f₋₂ isa Real
			@test f₋₁ isa Real

			# SemiRichardson
			@test_nowarn GRD.laurents_coeffs(K, el, û, x̂, GRD.SemiRichardsonExpansion())
			ℒ_semi = GRD.laurents_coeffs(K, el, û, x̂, GRD.SemiRichardsonExpansion())
			f₋₂, f₋₁ = ℒ_semi(π / 4)
			@test f₋₂ isa Real
			@test f₋₁ isa Real

			# Avec paramètres personnalisés
			params = GRD.RichardsonParams(atol = 1e-10, rtol = 1e-8, maxeval = 8)
			@test_nowarn GRD.laurents_coeffs(K, el, û, x̂, GRD.FullRichardsonExpansion(params))
		end

		@testset "guiggiani_singular_integral - basic" begin
			quad_rho = Inti.GaussLegendre(5)
			quad_theta = Inti.GaussLegendre(10)

			# Test avec méthode par défaut
			@test_nowarn GRD.guiggiani_singular_integral(K, û, x̂, el, quad_rho, quad_theta)
			I = GRD.guiggiani_singular_integral(K, û, x̂, el, quad_rho, quad_theta)
			@test I isa Real
			@test isfinite(I)
		end

		@testset "guiggiani_singular_integral - different methods" begin
			quad_rho = Inti.GaussLegendre(5)
			quad_theta = Inti.GaussLegendre(10)

			# Analytical
			@test_nowarn GRD.guiggiani_singular_integral(
				K, û, x̂, el, quad_rho, quad_theta, GRD.AnalyticalExpansion(),
			)
			I_anal = GRD.guiggiani_singular_integral(
				K, û, x̂, el, quad_rho, quad_theta, GRD.AnalyticalExpansion(),
			)
			@test I_anal isa Real
			@test isfinite(I_anal)

			# AutoDiff (avec SplitKernel)
			SK = GRD.SplitKernel(K)
			@test_nowarn GRD.guiggiani_singular_integral(
				SK, û, x̂, el, quad_rho, quad_theta, GRD.AutoDiffExpansion(),
			)
			I_ad = GRD.guiggiani_singular_integral(
				SK, û, x̂, el, quad_rho, quad_theta, GRD.AutoDiffExpansion(),
			)
			@test I_ad isa Real
			@test isfinite(I_ad)

			# FullRichardson avec paramètres personnalisés
			params = GRD.RichardsonParams(atol = 1e-10, maxeval = 8)
			@test_nowarn GRD.guiggiani_singular_integral(
				K, û, x̂, el, quad_rho, quad_theta, GRD.FullRichardsonExpansion(params),
			)
		end

		@testset "guiggiani_singular_integral - helper functions" begin
			# Test des fonctions helpers
			SK = GRD.SplitKernel(K)
			Kprod = (qx, qy) -> prod(SK(qx, qy))
			K_polar = GRD.polar_kernel_fun(Kprod, el, û, x̂)

			ρ = 0.1
			θ = π / 4
			f₋₂ = 1.0
			f₋₁ = 0.5

			# Test _regularized_integrand pour différents ordres
			@test_nowarn GRD._regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, Val(-2))
			@test_nowarn GRD._regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, Val(-1))
			@test_nowarn GRD._regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, Val(0))

			# Test _analytical_singular_contribution
			I_rho = 1.0
			ρ_max = 0.5
			@test_nowarn GRD._analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, Val(-2))
			@test_nowarn GRD._analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, Val(-1))
			@test_nowarn GRD._analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, Val(0))

			# Vérifier que les résultats sont des nombres
			result = GRD._regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, Val(-2))
			@test result isa Real

			result = GRD._analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, Val(-2))
			@test result isa Real
		end

		@testset "guiggiani_singular_integral - vectorial kernels" begin
			# Test avec un kernel élastique (résultat matriciel)
			K_elast = Inti.HyperSingularKernel(Inti.Elastostatic(μ = 1.0, λ = 1.0, dim = 3))
			quad_rho = Inti.GaussLegendre(5)
			quad_theta = Inti.GaussLegendre(10)

			@test_nowarn GRD.guiggiani_singular_integral(K_elast, û, x̂, el, quad_rho, quad_theta)
			I_elast = GRD.guiggiani_singular_integral(K_elast, û, x̂, el, quad_rho, quad_theta)
			@test I_elast isa AbstractMatrix
			@test all(isfinite, I_elast)
		end
	end
end
