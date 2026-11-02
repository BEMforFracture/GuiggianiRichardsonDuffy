using Test
import GuiggianiRichardsonDuffy as GRD
using Inti
using LinearAlgebra
using StaticArrays
using Random

# `prod(SplitKernel(K)(qx, qy))` must reproduce the kernel `K(qx, qy)` of Inti
# (singular part × regular part), for arbitrary points and normals.
@testset "SplitKernel consistency with Inti kernels" begin
	rng = Random.MersenneTwister(1234)
	ops = (
		Inti.Laplace(; dim = 2),
		Inti.Laplace(; dim = 3),
		Inti.Elastostatic(; dim = 3, μ = 1.3, λ = 0.7),
	)
	kernel_types = (
		Inti.SingleLayerKernel,
		Inti.DoubleLayerKernel,
		Inti.AdjointDoubleLayerKernel,
		Inti.HyperSingularKernel,
	)
	for op in ops, Kt in kernel_types
		N = Inti.ambient_dimension(op)
		@testset "$(nameof(typeof(op))){$N} - $(nameof(Kt))" begin
			K = Kt(op)
			SK = GRD.SplitKernel(K)
			s = Inti.singularity_order(K)
			ϵ = 0.0
			ϵ_sing = 0.0
			same_split = true
			for _ in 1:100
				qx = (coords = randn(rng, SVector{N, Float64}), normal = normalize(randn(rng, SVector{N, Float64})))
				qy = (coords = randn(rng, SVector{N, Float64}), normal = normalize(randn(rng, SVector{N, Float64})))
				ϵ = max(ϵ, norm(prod(SK(qx, qy)) - K(qx, qy)) / norm(K(qx, qy)))
				# the singular part is d^s (log d for the 2D single layer): AutoDiff and
				# SemiRichardson expansions discard it and rely on its sign
				d = norm(Inti.coords(qx) - Inti.coords(qy))
				sing_ref = (N == 2 && Kt === Inti.SingleLayerKernel) ? log(d) : d^s
				ϵ_sing = max(ϵ_sing, abs(first(SK(qx, qy)) - sing_ref) / abs(sing_ref))
				# an explicit direction r̂ = (x - y) / |x - y| gives the same split
				r̂ = (Inti.coords(qx) - Inti.coords(qy)) / d
				same_split &= all(map((a, b) -> isapprox(a, b; rtol = 1e-12), SK(qx, qy, r̂), SK(qx, qy)))
			end
			@test ϵ < 1e-12
			@test ϵ_sing < 1e-14
			@test same_split
		end
	end
end
