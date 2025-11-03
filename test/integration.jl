using Test
import GuiggianiRichardsonDuffy as GRD
using Inti
using Gmsh
using LinearAlgebra
using StaticArrays

ops = (
	Inti.Laplace(dim = 3),
	Inti.Elastostatic(; dim = 3, μ = 1.0, λ = 1.0),
)

testdir = @__DIR__  # Répertoire du fichier de test
projectdir = dirname(testdir)  # Répertoire racine du projet
mesh_path = joinpath(projectdir, "assets", "meshes_template", "disks", "disk_infinite_media.msh")
msh = GRD.@suppress_output Inti.import_mesh(mesh_path; dim = 3)
Γ_msh = view(msh, Inti.Domain(e -> "C" in Inti.labels(e), msh))

Q = Inti.Quadrature(Γ_msh; qorder = 2)

rich_params = GRD.RichardsonParams(atol = 1e-10, rtol = 1e-8, contract = 1 / 2, maxeval = 10, first_contract = 1 / 2)

methods = (
	GRD.FullRichardsonExpansion(rich_params),
	GRD.SemiRichardsonExpansion(rich_params),
	GRD.AutoDiffExpansion(),
	GRD.AnalyticalExpansion(),
)

@testset "Adaptive correction tests w.r.t Inti" begin
	for op in ops
		@testset "Operator: $(op)" begin
			for method in methods
				if method isa GRD.AnalyticalExpansion
					continue
				end
				@testset "Method: $(typeof(method))" begin
					K = Inti.HyperSingularKernel(op)
					iop = Inti.IntegralOperator(K, Q)
					δK = GRD.adaptive_correction(iop; method = method)
					δK_ref = Inti.adaptive_correction(iop)
					ϵ = norm(δK - δK_ref) / norm(δK_ref)
					@test ϵ < 1e-8
				end
			end
		end
	end
end

function _cod_laplace(radius, x::SVector{3, Float64}, F)::Float64
	r = min(norm(x), radius)
	return 4F / π * sqrt(radius^2 - r^2)
end

function _cod_elastostatic(x::SVector{3, Float64}, μ, λ, radius, F::SVector{3, Float64})::SVector{3, Float64} # only work for mode I (normal loading)
	r = min(norm(x), radius)
	ν = λ / (2 * (λ + μ))
	E = μ * (3 * λ + 2 * μ) / (λ + μ)
	COD = 8F * (1 - ν^2) / (π * E) * sqrt(radius^2 - r^2)
	return COD
end

rich_params = GRD.RichardsonParams(atol = 1e-10, rtol = 1e-8, contract = 1 / 2, maxeval = 10, first_contract = 1 / 2)

methods = (
	GRD.FullRichardsonExpansion(rich_params),
	GRD.SemiRichardsonExpansion(rich_params),
	GRD.AutoDiffExpansion(),
	GRD.AnalyticalExpansion(),
)

@testset "Screen in infinite media w.r.t closed form formula" begin
	@testset "Laplace" begin
		op = Inti.Laplace(dim = 3)
		K_op = Inti.HyperSingularKernel(op)
		iop = Inti.IntegralOperator(K_op, Q)
		for method in methods
			if method isa GRD.AnalyticalExpansion
				continue
			end
			@testset "Method: $(typeof(method))" begin
				δK = GRD.adaptive_correction(iop; method = method)
				K₀ = Inti.assemble_matrix(iop)
				K = K₀ + δK
				qvals = zeros(Float64, size(K, 2))
				F = 1.0
				rhs = [-F for _ in Q]
				ϕ_num = K \ rhs
				radius = 1.0
				cod_closed_form(x) = _cod_laplace(radius, x, F)
				ϕ = [cod_closed_form(q.coords) for q in Q]
				ϵ = norm(ϕ - ϕ_num, 2) / norm(ϕ, 2)
				@test ϵ < 3e-2
			end
		end
	end
	@testset "Elastostatic" begin
		μ = 1.0
		λ = 1.0
		op = Inti.Elastostatic(; dim = 3, μ = μ, λ = λ)
		K_op = Inti.HyperSingularKernel(op)
		iop = Inti.IntegralOperator(K_op, Q)
		for method in methods
			if method isa GRD.AnalyticalExpansion
				continue
			end
			@testset "Method: $(typeof(method))" begin
				δK = GRD.adaptive_correction(iop; method = method)
				K₀ = Inti.assemble_matrix(iop)
				_K = K₀ + δK
				n, m = size(_K)
				K = zeros(Float64, 3 * n, 3 * m)
				for i in 1:n, j in 1:m
					K[3(i-1)+1:3i, 3(j-1)+1:3j] = _K[i, j]
				end
				F = SVector(0.0, 0.0, 1.0)
				_rhs = [-F for _ in Q]
				rhs = reinterpret(Float64, _rhs)
				_ϕ_num = K \ rhs
				ϕ_num = reinterpret(SVector{3, Float64}, _ϕ_num)
				radius = 1.0
				cod_closed_form(x) = _cod_elastostatic(x, μ, λ, radius, F)
				ϕ = [cod_closed_form(q.coords) for q in Q]
				ϵ = norm(ϕ - ϕ_num, 2) / norm(ϕ, 2)
				@test ϵ < 3e-2
			end
		end
	end
end
