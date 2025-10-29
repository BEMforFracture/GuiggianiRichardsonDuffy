import GuiggianiRichardsonDuffy as GRD
using Inti
using StaticArrays
using BenchmarkTools

# Configuration identique
x̂ = SVector(0.5, 0.5)

# Quadrature
n_rho = 10
n_theta = 40
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

# Density function
û_vector = ŷ -> ones(SVector{1, Float64})  # Inti attend un vecteur
û_scalar = ξ -> 1.0  # Vous utilisez un scalaire

# Kernel
K = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))

# Orientation (requis par Inti)
ori = 1

println("="^70)
println("BENCHMARK COMPARATIF: Votre package vs Inti")
println("="^70)

# 1. Benchmark Inti natif
println("\n[1] Inti.guiggiani_singular_integral (natif)")
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)
b_inti = @benchmark Inti.guiggiani_singular_integral(
	$K, $û_vector, $x̂, $el, $ori, $quad_rho, $quad_theta,
) samples = 100 seconds = 5

println("  Time (median):   $(median(b_inti.times) / 1000) μs")
println("  Allocations:     $(median(b_inti.allocs))")
println("  Memory:          $(median(b_inti.memory) / 1000) kB")

# 2. Benchmark votre package (FullRichardson)
println("\n[2] GRD.guiggiani_singular_integral (FullRichardson)")
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 5,
)
method = GRD.FullRichardsonExpansion(rich_params)

b_yours = @benchmark GRD.guiggiani_singular_integral(
	$K, $û_scalar, $x̂, $el, $ori, $quad_rho, $quad_theta, $method,
) samples = 100 seconds = 5

println("  Time (median):   $(median(b_yours.times) / 1000) μs")
println("  Allocations:     $(median(b_yours.allocs))")
println("  Memory:          $(median(b_yours.memory) / 1000) kB")

# 3. Benchmark votre package (SemiRichardson)
println("\n[3] GRD.guiggiani_singular_integral (SemiRichardson)")
method_semi = GRD.SemiRichardsonExpansion(rich_params)

b_yours_semi = @benchmark GRD.guiggiani_singular_integral(
	$K, $û_scalar, $x̂, $el, $ori, $quad_rho, $quad_theta, $method_semi,
) samples = 100 seconds = 5

println("  Time (median):   $(median(b_yours_semi.times) / 1000) μs")
println("  Allocations:     $(median(b_yours_semi.allocs))")
println("  Memory:          $(median(b_yours_semi.memory) / 1000) kB")

# 4. Benchmark votre package (AutoDiff)
println("\n[4] GRD.guiggiani_singular_integral (AutoDiff)")
method_auto = GRD.AutoDiffExpansion()

b_yours_auto = @benchmark GRD.guiggiani_singular_integral(
	$K, $û_scalar, $x̂, $el, $ori, $quad_rho, $quad_theta, $method_auto,
) samples = 100 seconds = 5

println("  Time (median):   $(median(b_yours_auto.times) / 1000) μs")
println("  Allocations:     $(median(b_yours_auto.allocs))")
println("  Memory:          $(median(b_yours_auto.memory) / 1000) kB")

# 5. Benchmark votre package (Analytical)
println("\n[5] GRD.guiggiani_singular_integral (Analytical)")
method_analytical = GRD.AnalyticalExpansion()

b_yours_analytical = @benchmark GRD.guiggiani_singular_integral(
	$K, $û_scalar, $x̂, $el, $ori, $quad_rho, $quad_theta, $method_analytical,
) samples = 100 seconds = 5

println("  Time (median):   $(median(b_yours_analytical.times) / 1000) μs")
println("  Allocations:     $(median(b_yours_analytical.allocs))")
println("  Memory:          $(median(b_yours_analytical.memory) / 1000) kB")

# Comparaison
println("\n" * "="^70)
println("COMPARAISON")
println("="^70)

t_inti = median(b_inti.times) / 1000
t_yours_full = median(b_yours.times) / 1000
t_yours_semi = median(b_yours_semi.times) / 1000
t_yours_auto = median(b_yours_auto.times) / 1000
t_yours_anal = median(b_yours_analytical.times) / 1000

println("Inti (natif):              $(round(t_inti, digits=1)) μs")
println("Vous (FullRichardson):     $(round(t_yours_full, digits=1)) μs")
println("Vous (SemiRichardson):     $(round(t_yours_semi, digits=1)) μs")
println("Vous (AutoDiff):           $(round(t_yours_auto, digits=1)) μs")
println("Vous (Analytical):         $(round(t_yours_anal, digits=1)) μs")

println("\nRapports vs Inti:")
for (name, t) in [("FullRichardson", t_yours_full), ("SemiRichardson", t_yours_semi), ("AutoDiff", t_yours_auto), ("Analytical", t_yours_anal)]
	if t < t_inti
		ratio = (t_inti - t) / t_inti * 100
		println("  $name: ✅ $(round(ratio, digits=1))% plus rapide")
	else
		ratio = (t - t_inti) / t_inti * 100
		println("  $name: ⚠️  $(round(ratio, digits=1))% plus lent")
	end
end

println("\nClassement (du plus rapide au plus lent):")
times = [("Analytical", t_yours_anal), ("Inti", t_inti), ("SemiRichardson", t_yours_semi), ("AutoDiff", t_yours_auto), ("FullRichardson", t_yours_full)]
sort!(times, by = x -> x[2])
for (i, (name, t)) in enumerate(times)
	marker = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
	println("  $marker $i. $name: $(round(t, digits=1)) μs")
end

println("="^70)
