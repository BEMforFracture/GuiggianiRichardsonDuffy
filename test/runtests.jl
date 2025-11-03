using Test
import GuiggianiRichardsonDuffy as GRD

@testset "Smoke tests" begin
	include("smoketests.jl")
end

@testset "Unit tests" begin
	for file in filter(f -> startswith(f, "test-") && endswith(f, ".jl"), readdir(@__DIR__))
		include(file)
	end
end

@testset "Integration tests" begin
	include("integration.jl")
end
