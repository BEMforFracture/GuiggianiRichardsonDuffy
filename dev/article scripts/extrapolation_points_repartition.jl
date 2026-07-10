"""
gl_vs_geometric_extrapolation.jl

Compare la precision d'extrapolation du coefficient de Laurent F_{-1}
(type Guiggiani/Richardson) selon que les points d'evaluation rho_i
sont pris :

  (A) aux noeuds de Gauss-Legendre, mappes lineairement sur [0, rho_hat] ;
  (B) sur une suite geometrique dediee, resserree pres de rho = 0.

Fonction test : G(rho) = 1/(1+rho), avec valeurs exactes connues
    F_{-2} = G(0)  = 1
    F_{-1} = G'(0) = -1
et rayon de convergence = 1 (cas defavorable : la "singularite" du
test est juste au bord du domaine radial rho_hat = 1, pour illustrer
le pire des cas).

Utilise Inti.GaussLegendre(n) pour les noeuds/poids de Gauss-Legendre
(deja present dans l'environnement de travail), plutot que
FastGaussQuadrature. Inti.GaussLegendre(n)() renvoie des noeuds/poids
sur [0, 1] ; on les ramene sur [-1, 1] par x = 2*x01 - 1 (poids x2).

Dependances (a installer si besoin) :
    ] add Polynomials Plots
"""

using Inti
using Polynomials
using Plots
using GuiggianiRichardsonDuffy
using StaticArrays

# ------------------------------------------------------------------
# Fonction test et valeurs exactes
# ------------------------------------------------------------------

θ = 0.0

y1 = SVector(0.0, 0.0, 0.0)
y2 = SVector(1.0, 0.0, 0.0)
y3 = SVector(0.0, 1.0, 0.0)
y4 = SVector(1.0, 1.0, 0.0)

el = Inti.LagrangeSquare(y1, y2, y3, y4)

op = Inti.Laplace(; dim = 3)
K = Inti.HyperSingularKernel(op)
û = ξ -> 1.0
x̂ = SVector(0.5, 0.5)
ori = 1
F = GuiggianiRichardsonDuffy.polar_kernel_fun(K, el, û, x̂, ori)
G(ρ) = ρ^2 * F(ρ, θ)
const F2_exact = 1.0
const F1_exact = -1.0
const rho_hat  = 1.0

# ------------------------------------------------------------------
# Extrapolation de Lagrange : renvoie (P(0), P'(0)) ~ (F_{-2}, F_{-1})
# ------------------------------------------------------------------
"""
    extrapolate(rho) -> (F2, F1)

Construit le polynome interpolateur exact (degre length(rho)-1) passant
par (rho_i, G(rho_i)), puis evalue P(0) et P'(0).
"""
function extrapolate(rho::AbstractVector{<:Real})
    y  = G.(rho)
    p  = fit(rho, y)          # interpolation exacte, degre length(rho)-1
    dp = derivative(p)
    return p(0.0), dp(0.0)
end

# ------------------------------------------------------------------
# Generateurs de sequences de points radiaux
# ------------------------------------------------------------------
"""
    gl_nodes_pm1(n)

Noeuds et poids de Gauss-Legendre sur [-1, 1]. Inti.GaussLegendre(n)()
renvoie des noeuds/poids sur [0, 1] ; changement de variable affine
t = 2x - 1 (jacobien dt = 2 dx, donc poids multiplies par 2).
"""
function gl_nodes_pm1(n::Int)
    x01, w01 = Inti.GaussLegendre(n)()
    x01 = collect([_x[1] for _x in x01])
    x = 2.0 .* x01 .- 1.0
    w = 2.0 .* w01
    return x, w
end

function gl_nodes(n::Int, rho_hat::Real)
    x, _ = gl_nodes_pm1(n)
    rho  = (rho_hat / 2) .* (1.0 .+ x)      # mappage lineaire sur [0, rho_hat]
    return sort(rho)
end

geometric_nodes(n::Int; rho0::Real = 0.5, contract::Real = 0.5) =
    rho0 .* contract .^ (0:n-1)

# ------------------------------------------------------------------
# Balayage en n et calcul des erreurs sur F_{-1}
# ------------------------------------------------------------------
ns      = 2:15
err_gl  = Float64[]
err_geo = Float64[]

for n in ns
    _, F1_gl  = extrapolate(gl_nodes(n, rho_hat))
    _, F1_geo = extrapolate(geometric_nodes(n))
    push!(err_gl,  abs(F1_gl  - F1_exact))
    push!(err_geo, abs(F1_geo - F1_exact))
end

clip(x) = max(x, eps())   # plancher a la precision machine (pour l'echelle log)

# ------------------------------------------------------------------
# Trace des courbes de convergence
# ------------------------------------------------------------------
plt = plot(collect(ns), clip.(err_gl);
    yscale    = :log10,
    marker    = :circle,
    linewidth = 2,
    label     = "Noeuds Gauss-Legendre",
    xlabel    = "nombre de points n",
    ylabel    = "erreur absolue sur F₋₁",
    title     = "Extrapolation du coefficient de Laurent F₋₁",
    legend    = :topright,
    dpi       = 300,
)
plot!(plt, collect(ns), clip.(err_geo);
    marker    = :utriangle,
    linestyle = :dash,
    linewidth = 2,
    label     = "Suite geometrique (rho0=0.5, contract=0.5)",
)

savefig(plt, joinpath(@__DIR__, "gl_vs_geometric_extrapolation.pdf"))
savefig(plt, joinpath(@__DIR__, "gl_vs_geometric_extrapolation.png"))

println("Figure enregistree dans : ", @__DIR__)
println()
println("n   |  erreur GL   |  erreur geometrique")
for (n, eg, ee) in zip(ns, err_gl, err_geo)
    println(rpad(n, 4), "|  ", eg, "  |  ", ee)
end
