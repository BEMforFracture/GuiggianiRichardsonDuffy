# Guide d’utilisation

## Choix de la méthode d’expansion des coefficients de Laurent

- `:analytical` (si disponible): plus rapide et précis.
- `:auto_diff`: nécessite un noyau translation-invariant, calcule via AD.
- `:semi_richardson`: F₋₂ analytique, F₋₁ par Richardson.
- `:full_richardson`: générique, pas d’hypothèse forte.

Exemple :

```julia
I = guiggiani_singular_integral(
    GuiggianiRichardsonDuffy.SplitLaplaceHypersingular,
    ξ -> 1.0,
    SVector(0.2, 0.3),
    Inti.LagrangeSquare((SVector(0.0,0.0,0.0), SVector(1.0,0.0,0.0),
                          SVector(0.0,1.0,0.0), SVector(1.0,1.0,0.0))),
    16, 32;
    expansion = :full_richardson,
    first_contract = 1e-2, contract = 0.5,
)
```

## Réglages de quadrature

- `n_rho`, `n_theta` contrôlent la précision angulaire/radiale.
- Des valeurs typiques: 8–32 selon la régularité de `û` et la courbure locale.

## Paramètres des noyaux élastostatiques

Passer `λ, μ` via les mots-clés :

```julia
K = GuiggianiRichardsonDuffy.SplitElastostaticHypersingular
I = guiggiani_singular_integral(K, ξ->1.0, SVector(0.3,0.4), el, 16, 32; expansion=:analytical, λ=1.0, μ=0.5)
```

## Aide

Depuis le REPL, `?guiggiani_singular_integral` pour la docstring détaillée.