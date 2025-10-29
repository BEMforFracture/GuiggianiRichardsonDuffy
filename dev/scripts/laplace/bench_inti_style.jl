import GuiggianiRichardsonDuffy as GRD
using Inti
using StaticArrays
using BenchmarkTools

# Configuration
x̂ = SVector(0.5, 0.5)

rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 5,
)

n_rho = 10
n_theta = 40

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

û = ξ -> 1.0
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))

println("="^70)
println("BENCHMARK: Inti-style Optimization (FullRichardson only)")
println("="^70)

method = GRD.FullRichardsonExpansion(rich_params)

println("\n[FullRichardson - Inti-style]")

# Benchmark
b = @benchmark GRD.guiggiani_singular_integral(
	$K_base, $û, $x̂, $el, $n_rho, $n_theta, $method,
) samples = 100 seconds = 5

println("  Time (median):   $(median(b.times) / 1000) μs")
println("  Allocations:     $(median(b.allocs))")
println("  Memory:          $(median(b.memory) / 1000) kB")

println("\n" * "="^70)
println("Comparison with previous version (perf/optimize):")
println("  perf/optimize:   474 μs, 4724 allocs, 299 kB")
println("  Inti-style:      $(median(b.times) / 1000) μs, $(median(b.allocs)) allocs, $(median(b.memory) / 1000) kB")
if median(b.times) / 1000 < 474
	gain = (474 - median(b.times) / 1000) / 474 * 100
	println("  --> ✅ $(round(gain, digits=1))% faster!")
else
	loss = (median(b.times) / 1000 - 474) / 474 * 100
	println("  --> ⚠️  $(round(loss, digits=1))% slower")
end
println("="^70)
