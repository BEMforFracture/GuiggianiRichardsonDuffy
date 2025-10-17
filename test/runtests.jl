using Pkg: Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));

using Test
import GuiggianiRichardsonDuffy as GRD

for file in filter(f -> startswith(f, "test-") && endswith(f, ".jl"), readdir(@__DIR__))
	include(file)
end
