import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using BenchmarkTools

# INPUTS

x̂ = SVector(0.5, 0.5) # source point in reference coordinates

### Richardson extrapolation parameters
maxeval = 10
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

# quadrature parameters
n_rho = 5
n_theta = 10

# END INPUTS

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

K = GRD.SplitLaplaceHypersingular

b_dict_gui = Dict{Symbol, BenchmarkTools.Trial}()

for method in GRD.EXPANSION_METHODS
	b = @benchmark begin
		res = GRD.guiggiani_singular_integral(
			$K,
			$û,
			$x̂,
			$el,
			$n_rho,
			$n_theta;
			sorder = Val(-2),
			expansion = $method,
			rtol = $rtol,
			maxeval = $maxeval,
			first_contract = $first_contract,
			breaktol = $breaktol,
			contract = $contract,
			atol = $atol,
		)
	end
	b_dict_gui[method] = b
end

for (method, b) in b_dict_gui
	println("Singular integral, ", method, ":")
	display(b)
	println()
end

b_dict_laurent = Dict{Symbol, BenchmarkTools.Trial}()

for method in GRD.EXPANSION_METHODS
	b = @benchmark begin
		F₋₂, F₋₁ = GRD.laurents_coeffs(
			$K,
			$el,
			$û,
			$x̂;
			expansion = $method,
			maxeval = $maxeval,
			rtol = $rtol,
			atol = $atol,
			contract = $contract,
			first_contract = $first_contract,
			breaktol = $breaktol,
		)
	end
	b_dict_laurent[method] = b
end

for (method, b) in b_dict_laurent
	println("Laurent coefficients, ", method, ":")
	display(b)
	println()
end

b_dict_eval = Dict{Symbol, BenchmarkTools.Trial}()
# I want to benchmark the evaluation of F₋₂(θ) and F₋₁(θ) at random values of θ
θ_test = 2π * rand(n_theta)

for method in GRD.EXPANSION_METHODS
	F₋₂, F₋₁ = GRD.laurents_coeffs(
		K,
		el,
		û,
		x̂;
		expansion = method,
		maxeval = maxeval,
		rtol = rtol,
		atol = atol,
		contract = contract,
		first_contract = first_contract,
		breaktol = breaktol,
	)
	b = @benchmark begin
		for θ in $θ_test
			$F₋₂(θ)
			$F₋₁(θ)
		end
	end
	b_dict_eval[method] = b
end

for (method, b) in b_dict_eval
	println("Laurent coefficients evaluation, ", method, ":")
	display(b)
	println()
end
