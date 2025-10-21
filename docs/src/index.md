# GuiggianiRichardsonDuffy.jl

A Julia package for the numerical computation of laurent's coefficients expansion of singular kernels, and the evaluation of the associated singular integrals over boundary elements, which consist of computing the Cauchy principal value or Hadamard finite part over parametric elements.

The idea of this package was originally proposed by Guiggiani in 1992 [guiggianiGeneralAlgorithmNumerical1992](@cite).

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
- Docstrings are available in the [Docstrings](docstrings.md) section.
