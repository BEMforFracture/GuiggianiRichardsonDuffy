using Test
import GuiggianiRichardsonDuffy as GRD
using Inti
using LinearAlgebra
using StaticArrays

# AutoDiff and SemiRichardson build the Laurent coefficients from the split kernel, evaluated along the
# direction of y - x; FullRichardson extrapolates the kernel itself. A sign or factor mismatch between the
# split kernel and the direction used for the coefficients (e.g. r̂ = (x - y)/|x - y| vs Â ∝ y - x for the odd
# kernels DL and ADL) leaves the split kernels themselves correct (test-split-kernels.jl passes) but corrupts
# f₋₁ and hence the singular integrals. This test compares, on one flat and one curved element, the
# coefficients f₋₁(θ) and the singular integrals of the three methods.
@testset "Expansion methods consistent with FullRichardson (elastostatic 3D)" begin
	op = Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)
	rp = GRD.RichardsonParams(atol = 1e-12, rtol = 1e-10, contract = 1 / 2, maxeval = 12, first_contract = 1 / 2)
	quad = Inti.LagrangeSquare((SVector(0.0, 0, 0), SVector(1.0, 0.1, 0), SVector(0.2, 0.9, 0), SVector(1.3, 1.2, 0)))
	sph(p) = normalize(SVector(p...))
	v1, v2, v3 = sph((1, 0.1, 0.1)), sph((0.1, 1, 0.1)), sph((0.1, 0.1, 1))
	tri6 = Inti.LagrangeTriangle((v1, v2, v3, sph((v1 + v2) / 2), sph((v2 + v3) / 2), sph((v1 + v3) / 2)))
	û = ξ -> 1.0
	kernels = (("SL", Inti.SingleLayerKernel(op)), ("DL", Inti.DoubleLayerKernel(op)),
		("ADL", Inti.AdjointDoubleLayerKernel(op)), ("HS", Inti.HyperSingularKernel(op)))
	for (ename, el, x̂) in (("flat quadrangle", quad, SVector(0.3, 0.6)), ("curved P2 triangle", tri6, SVector(0.2, 0.3)))
		for (kname, K) in kernels
			@testset "$ename, $kname" begin
				for θ in (0.4, 2.5)
					cF = GRD.laurents_coeffs(K, el, 1, û, x̂, GRD.FullRichardsonExpansion(rp))(θ)
					for m in (GRD.AutoDiffExpansion(), GRD.SemiRichardsonExpansion(rp))
						c = GRD.laurents_coeffs(K, el, 1, û, x̂, m)(θ)
						for k in 1:2
							nF = norm(cF[k])
							nF > 1e-10 && @test norm(c[k] - cF[k]) / nF < 1e-6
						end
					end
				end
				IF = GRD.guiggiani_singular_integral(K, û, x̂, el, 1, Inti.GaussLegendre(30), Inti.GaussLegendre(60), GRD.FullRichardsonExpansion(rp))
				for m in (GRD.AutoDiffExpansion(), GRD.SemiRichardsonExpansion(rp))
					I = GRD.guiggiani_singular_integral(K, û, x̂, el, 1, Inti.GaussLegendre(30), Inti.GaussLegendre(60), m)
					@test norm(I - IF) / norm(IF) < 1e-6
				end
			end
		end
	end
end
