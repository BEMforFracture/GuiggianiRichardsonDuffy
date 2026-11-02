using Test
import GuiggianiRichardsonDuffy as GRD
using Inti
using Gmsh
using LinearAlgebra
using StaticArrays
using Logging

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

# Singular corrections (maxdist = 0: only the element containing the target node) computed by
# GRD for the four elastostatic kernels and the three expansion methods, compared with
# `Inti.adaptive_correction`, on a flat mesh (P1 quadrangles) and on a curved mesh (P2 sphere).
# The reference uses Inti's adaptive polar quadrature with rtol = 1e-6 (agrees to ~1e-9 with GRD
# on a refined GaussLegendre(20) × GaussLegendre(60) polar rule). FullRichardsonExpansion evaluates
# the integrand with the kernel itself (the split kernel only enters its Laurent coefficients) and
# serves as a witness: the tolerances are set from its own error w.r.t. Inti, which is dominated by
# GRD's fixed polar quadrature (GaussLegendre(5) × GaussLegendre(20)): measured FullRichardson
# errors are about 1e-9 (SL), 2e-8 (DL, ADL), 8e-8 (HS) on the disk and 1e-5 (SL), 1e-3 (DL, ADL),
# 2e-3 (HS) on the sphere; the tolerances are about 10 times these values.

function _grd_test_disk_mesh()
	mesh_path = joinpath(dirname(@__DIR__), "assets", "meshes_template", "disks", "disk_infinite_media.msh")
	Inti.clear_entities!()
	msh = GRD.@suppress_output Inti.import_mesh(mesh_path; dim = 3)
	return view(msh, Inti.Domain(e -> "C" in Inti.labels(e), msh))
end

function _grd_test_sphere_mesh(h)
	Inti.clear_entities!()
	gmsh.initialize(String[], false)
	gmsh.option.setNumber("General.Verbosity", 0)
	gmsh.model.occ.addSphere(0, 0, 0, 1)
	gmsh.model.occ.synchronize()
	gmsh.option.setNumber("Mesh.MeshSizeMin", h)
	gmsh.option.setNumber("Mesh.MeshSizeMax", h)
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.setOrder(2)
	msh = Inti.import_mesh(; dim = 3)
	gmsh.finalize()
	return view(msh, Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh))
end

@testset "Singular correction w.r.t. Inti (elastostatic, flat and curved)" begin
	op = Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)
	kernels = (
		SL = Inti.SingleLayerKernel(op),
		DL = Inti.DoubleLayerKernel(op),
		ADL = Inti.AdjointDoubleLayerKernel(op),
		HS = Inti.HyperSingularKernel(op),
	)
	rich_params = GRD.RichardsonParams(atol = 1e-10, rtol = 1e-8, contract = 1 / 2, maxeval = 10, first_contract = 1 / 2)
	methods = (
		GRD.FullRichardsonExpansion(rich_params),
		GRD.SemiRichardsonExpansion(rich_params),
		GRD.AutoDiffExpansion(),
	)
	meshes = (
		(name = "flat disk (P1 quadrangles)", Γ = _grd_test_disk_mesh(), tol = (SL = 1e-8, DL = 2e-7, ADL = 2e-7, HS = 1e-6)),
		(name = "sphere (P2 triangles)", Γ = _grd_test_sphere_mesh(0.8), tol = (SL = 1e-4, DL = 1e-2, ADL = 1e-2, HS = 2e-2)),
	)
	for m in meshes
		Q = Inti.Quadrature(m.Γ; qorder = 2)
		@testset "$(m.name)" begin
			for (kname, K) in pairs(kernels)
				iop = Inti.IntegralOperator(K, Q)
				δK_ref = _quiet(() -> Inti.adaptive_correction(iop; maxdist = 0.0, rtol = 1e-6, atol = 0.0))
				δK_full = nothing
				for method in methods
					@testset "$kname - $(nameof(typeof(method)))" begin
						δK = _quiet(() -> GRD.adaptive_correction(iop; method = method, maxdist = 0.0))
						ϵ = norm(δK - δK_ref) / norm(δK_ref)
						@test ϵ < m.tol[kname]
						# same polar quadrature as FullRichardson: only the Laurent coefficients and
						# the (split) integrand differ, so the agreement must be much tighter
						if method isa GRD.FullRichardsonExpansion
							δK_full = δK
						else
							@test norm(δK - δK_full) / norm(δK_full) < 1e-6
						end
					end
				end
			end
		end
	end
end

# Laurent coefficients of the polar integrand: AutoDiff and SemiRichardson evaluate the regular
# part of the split kernel in the limit direction, FullRichardson extrapolates the kernel itself.
# The odd kernels (DL, ADL) detect a direction or sign inconsistency between both paths.
@testset "Laurent coefficients: AutoDiff/SemiRichardson vs FullRichardson (curved P2 element)" begin
	op = Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0)
	rich_params = GRD.RichardsonParams(atol = 1e-12, rtol = 1e-10, contract = 1 / 2, maxeval = 12, first_contract = 1 / 2)
	sph(p) = normalize(SVector{3, Float64}(p...))
	v1, v2, v3 = sph((1, 0.1, 0.1)), sph((0.1, 1, 0.1)), sph((0.1, 0.1, 1))
	el = Inti.LagrangeTriangle((v1, v2, v3, sph((v1 + v2) / 2), sph((v2 + v3) / 2), sph((v1 + v3) / 2)))
	x̂ = SVector(0.2, 0.3)
	û = ξ -> 1.0
	for K in (Inti.DoubleLayerKernel(op), Inti.AdjointDoubleLayerKernel(op), Inti.HyperSingularKernel(op))
		@testset "$(nameof(typeof(K)))" begin
			ℒ_full = GRD.laurents_coeffs(K, el, 1, û, x̂, GRD.FullRichardsonExpansion(rich_params))
			ℒ_ad = GRD.laurents_coeffs(K, el, 1, û, x̂, GRD.AutoDiffExpansion())
			ℒ_semi = GRD.laurents_coeffs(K, el, 1, û, x̂, GRD.SemiRichardsonExpansion(rich_params))
			for θ in (0.4, 2.5, 4.0)
				f_full, f_ad, f_semi = ℒ_full(θ), ℒ_ad(θ), ℒ_semi(θ)
				# leading coefficient (f₋₁ for DL/ADL, f₋₂ for HS)
				k = Inti.singularity_order(K) == -3 ? 1 : 2
				@test norm(f_ad[k] - f_full[k]) / norm(f_full[k]) < 1e-8
				@test norm(f_semi[k] - f_full[k]) / norm(f_full[k]) < 1e-8
			end
		end
	end
end
