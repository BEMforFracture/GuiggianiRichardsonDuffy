import .GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
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

fig1 = Figure(; size = (1200, 800))

n_rho = 10
n_theta = 10

method = :full_richardson

b_dict = Dict{Symbol, BenchmarkTools.Trial}()

for method in GRD.EXPANSION_METHODS
	b = @benchmark begin
		res = GRD.guiggiani_singular_integral(
			K,
			û,
			x̂,
			el,
			n_rho,
			n_theta;
			sorder = Val(-2),
			expansion = method,
			rtol = rtol,
			maxeval = maxeval,
			first_contract = first_contract,
			breaktol = breaktol,
			contract = contract,
			atol = atol,
		)
	end
	b_dict[method] = b
end


