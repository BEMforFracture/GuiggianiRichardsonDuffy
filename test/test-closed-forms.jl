using Inti
using StaticArrays

@testset "Closed forms Laplace hypersingular integration on plane element" begin
	δ = 0.5
	z = 0.0
	y¹ = SVector(-1.0, -1.0, z)
	y² = SVector(1.0 + δ, -1.0, z)
	y³ = SVector(-1.0, 1.0, z)
	y⁴ = SVector(1.0 - δ, 1.0, z)
	nodes = (y¹, y², y³, y⁴)
	el = Inti.LagrangeSquare(nodes)
	a = SVector(0.0, 0.0, 0.0)
	b = SVector(0.66, 0.0, 0.0)
	c = SVector(0.479226, 0.66, 0.0)
	va = GRD.hypersingular_laplace_integral_on_plane_element(a, el)
	vb = GRD.hypersingular_laplace_integral_on_plane_element(b, el)
	vc = GRD.hypersingular_laplace_integral_on_plane_element(c, el)
	@test isapprox(va, -5.749237 / (4π); atol = 1.0e-5)
	@test isapprox(vb, -9.154585 / (4π); atol = 1.0e-5)
	@test isapprox(vc, -15.32850 / (4π); atol = 1.0e-5)
end
