using BenchmarkTools
using Inti
using LinearAlgebra
using StaticArrays
import GuiggianiRichardsonDuffy as GRD

# Configuration
pde = Inti.Laplace(; dim = 3)
K = Inti.SingleLayerKernel(pde)
SK = GRD.split_kernel(K)

# Créer des points de quadrature (named tuples avec coords et normal)
target = (coords = SVector(0.5, 0.5, 0.5), normal = SVector(0.0, 0.0, 1.0))
source = (coords = SVector(1.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))

println("=" ^ 80)
println("Benchmark du surcoût potentiel dans SplitKernel")
println("=" ^ 80)

# Test 1: Mesurer le coût de Inti.singularity_order
println("\n1. Benchmark de Inti.singularity_order(K)")
println("-" ^ 80)
@btime Inti.singularity_order($K)

# Test 2: Mesurer le coût de Inti.return_type
println("\n2. Benchmark de Inti.return_type(K, typeof(target), typeof(source))")
println("-" ^ 80)
@btime Inti.return_type($K, $(typeof(target)), $(typeof(source)))

# Test 3: Version optimisée avec pré-calcul
println("\n3. Benchmark de SplitKernel sans pré-calcul (version actuelle)")
println("-" ^ 80)
@btime $SK($target, $source)

# Test 4: Version optimisée avec pré-calcul des constantes
println("\n4. Benchmark de SplitKernel avec pré-calcul")
println("-" ^ 80)

# Simuler une version optimisée où s et T sont pré-calculés
function optimized_splitkernel(SK, target, source)
    K = SK.kernel
    # Ces valeurs sont pré-calculées (constantes du type)
    s = -1  # Inti.singularity_order(K)
    T = Float64  # Inti.return_type(K, typeof(target), typeof(source))
    
    r = Inti.coords(target) - Inti.coords(source)
    d = norm(r)
    
    if d == 0
        singular_part = zero(T)
        _, regular_part = GRD._extract_split_parts(K, target, source, nothing, d, s)
    else
        r̂ = r / d
        singular_part, regular_part = GRD._extract_split_parts(K, target, source, r̂, d, s)
    end
    
    return (singular_part, regular_part)
end

@btime optimized_splitkernel($SK, $target, $source)

# Test 5: Comparaison en boucle (scénario réaliste)
println("\n5. Benchmark en boucle (1000 appels)")
println("-" ^ 80)

n_iter = 1000

println("Version actuelle:")
@btime for i in 1:$n_iter
    $SK($target, $source)
end

println("\nVersion optimisée:")
@btime for i in 1:$n_iter
    optimized_splitkernel($SK, $target, $source)
end

println("\n" * "=" ^ 80)
println("Conclusion:")
println("Comparez les temps de la version actuelle (test 3) vs optimisée (test 4)")
println("Si la différence est significative, cela confirme l'hypothèse du surcoût.")
println("=" ^ 80)
