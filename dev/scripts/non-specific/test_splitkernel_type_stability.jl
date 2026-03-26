using Inti
using StaticArrays
import GuiggianiRichardsonDuffy as GRD

# Configuration
pde = Inti.Laplace(; dim = 3)
K = Inti.SingleLayerKernel(pde)
SK = GRD.SplitKernel(K)

# Points de test
target = (coords = SVector(0.5, 0.5, 0.5), normal = SVector(0.0, 0.0, 1.0))
source = (coords = SVector(1.0, 0.0, 0.0), normal = SVector(0.0, 0.0, 1.0))

println("=" ^ 80)
println("Test de stabilité de type pour SplitKernel")
println("=" ^ 80)

println("\n1. Type de SK:")
println("   ", typeof(SK))

println("\n2. Type de SK.kernel:")
println("   ", typeof(SK.kernel))

println("\n3. Type de SK.singularity_order:")
println("   ", typeof(SK.singularity_order))

println("\n4. Valeur de SK.singularity_order:")
println("   ", SK.singularity_order)

println("\n5. Test d'appel de SK:")
result = SK(target, source)
println("   Type du résultat: ", typeof(result))
println("   Valeur: ", result)

println("\n6. Test de code_warntype:")
using InteractiveUtils
println("\n@code_warntype SK(target, source):")
@code_warntype SK(target, source)

println("\n=" ^ 80)
