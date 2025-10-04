import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using Plots

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

x̂ = SVector(0.5, 0.5)

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)

p = 1
û = ξ -> Inti.lagrange_basis(typeof(el))(ξ)[p]

F₋₂ = θ -> GRD.LaplaceHypersingularClosedFormF₋₂(θ, x̂, el, û)
F₋₁ = θ -> GRD.LaplaceHypersingularClosedFormF₋₁(θ, x̂, el, û)

K = GRD.SplitLaplaceHypersingular
F̃₋₂ = GRD.f_minus_two_func(K, el, û, x̂)

Km = GRD.LaplaceHypersingular
K_polar = GRD.polar_kernel_fun(Km, el, û, x̂)
rho_max_fun = GRD.rho_fun(ref_domain, x̂)

F̃₋₁ = GRD.f_minus_one_func(K_polar, rho_max_fun, F̃₋₂; first_contract = 1e-2, contract = 0.5)

N = 10000
θs = range(0, 2π, length = N)

plot(θs, F₋₂.(θs), label = "F₋₂ exact", lw = 2)
plot!(θs, F̃₋₂.(θs), label = "F₋₂ exact bis", lw = 2, ls = :dash)
plot!(θs, F₋₁.(θs), label = "F₋₁ exact", lw = 2)
plot!(θs, F̃₋₁.(θs), label = "F₋₁ approx", lw = 2, ls = :dash)
xlabel!("θ")
savefig("./dev/figures/hybrid_richardson_vs_closed_form.png")
