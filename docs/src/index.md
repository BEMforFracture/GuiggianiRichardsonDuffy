# GuiggianiRichardsonDuffy.jl

TODO : resume 

## Install


## Quickstart

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

- See the complete [Guide](guide.md) for variants and parameters.
- The complete API documentation is available in the [API](api.md) section.
- Dosctrings are available in the [Docstrings](dosctrings.md) section.