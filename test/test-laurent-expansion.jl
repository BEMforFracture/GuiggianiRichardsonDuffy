using StaticArrays
using LinearAlgebra

# INPUTS

x̂ = SVector(0.4, 0.6) # source point in reference coordinates

### Richardson extrapolation parameters

maxeval = 10
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

# END INPUTS

kwargs_rich = (maxeval = maxeval, rtol = rtol, atol = atol, contract = contract, first_contract = first_contract, breaktol = breaktol)


methods = GRD.EXPANSION_METHODS
methods = [method for method in methods if method != :analytical]
kernels = GRD.ANALYTICAL_KERNELS

functions = (ξ -> 1.0, ξ -> ξ[1], ξ -> ξ[2] * ξ[1], ξ -> ξ[1]^2 - ξ[2])

kernels_dict = Dict{Symbol, Any}()

kernels_dict[:LaplaceHypersingular] = GRD.SplitLaplaceHypersingular
kernels_dict[:ElastostaticHypersingular] = GRD.SplitElastostaticHypersingular

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

@testset "Laurent Coefficients" begin
	@testset "Laplace Hypersingular Kernel" begin
		for method in methods
			for (kernel_name, K) in kernels_dict
				@testset "$method" begin
					for fun in functions
						û = fun
						kwargs_kernel = kernel_name == :LaplaceHypersingular ? (;) : (; λ = 1.0, μ = 1.0)
						D = Dict{Symbol, Tuple{Function, Function}}()
						F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :analytical, name = kernel_name, kernel_kwargs = kwargs_kernel)
						D[:analytical] = (F₋₂, F₋₁)
						if method == :auto_diff
							F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = method, kernel_kwargs = kwargs_kernel)
						else
							F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = method, name = kernel_name, kernel_kwargs = kwargs_kernel, richardson_kwargs = kwargs_rich)
						end
						D[method] = (F₋₂, F₋₁)

						N_θ = 100
						θs = range(0, 2π; length = N_θ)

						error_F₋₂ = norm(D[:analytical][1].(θs) - D[method][1].(θs), 2) / norm(D[:analytical][1].(θs), 2)
						error_F₋₁ = norm(D[:analytical][2].(θs) - D[method][2].(θs), 2) / norm(D[:analytical][2].(θs), 2)

						@test error_F₋₂ < 1e-11
						@test error_F₋₁ < 1e-6
					end
				end
			end
		end
	end
end
