# GuiggianiRichardsonDuffy.jl

Outils pour l’intégration précise d’intégrales singulières/hypersingulières en BEM
(Laplace, Élastostatique) via l’algorithme de Guiggiani, avec coefficients de Laurent
calculés analytiquement, par AD, ou par extrapolation de Richardson.

## Installation

Installez le package (en mode dev si local), puis Documenter pour construire ce site.

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path = ".")
Pkg.activate("docs")
Pkg.instantiate()
```

Pour construire la documentation :

```julia
julia --project=docs docs/make.jl
```

## Démarrage rapide

```julia
using Inti, StaticArrays
using GuiggianiRichardsonDuffy

el = Inti.LagrangeSquare((SVector(0.0,0.0,0.0), SVector(1.0,0.0,0.0),
                          SVector(0.0,1.0,0.0), SVector(1.0,1.0,0.0)))
x̂ = SVector(0.3, 0.4)
K = GuiggianiRichardsonDuffy.SplitLaplaceHypersingular
û(ξ) = 1.0

I = guiggiani_singular_integral(K, û, x̂, el, 16, 32; expansion = :full_richardson)
```

- Voir le [Guide](guide.md) pour les variantes et paramètres.
- La référence complète est dans la section [API](api.md).