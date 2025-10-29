using Test

dir = @__DIR__
scripts_dir = normpath(abspath(joinpath(dir, "..", "scripts")))  # chemin normalisé (plus de "..")

@testset "dev tests" begin
	for dir in readdir(scripts_dir)
		for file in filter(f -> endswith(f, ".jl"), readdir(joinpath(scripts_dir, dir)))
			include(joinpath(scripts_dir, dir, file))
		end
	end
end
