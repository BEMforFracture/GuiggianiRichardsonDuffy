using Inti
using GuiggianiRichardsonDuffy
using LinearAlgebra

hs = [100.0, 50.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

n_fixed_rho = 2
n_fixed_theta = 2

op = Inti.Elastostatic(λ = 1.0, μ = 1.0; dim = 3)
K = Inti.HyperSingularKernel(op)
û = ξ -> 1.0
quad_rho = Inti.GaussLegendre(n_fixed_rho)
quad_theta = Inti.GaussLegendre(n_fixed_theta)

x̂ = SVector(0.5, 0.5)

function build_element(h)
    y1 = SVector(0.0, 0.0, 0.0)
    y2 = SVector(h, 0.0, 0.0)
    y3 = SVector(0.0, h, 0.0)
    y4 = SVector(h, h, 0.0)

    el = Inti.LagrangeSquare(y1, y2, y3, y4)
    return el
end

for h in hs
    el = build_element(h)
    x = el(x̂)
    # I_exact = GuiggianiRichardsonDuffy.hypersingular_laplace_integral_on_plane_element(x, el)
    I_exact = Inti.guiggiani_singular_integral(K, û, x̂, el, 1, Inti.GaussLegendre(20), Inti.GaussLegendre(40))
    I_num = Inti.guiggiani_singular_integral(K, û, x̂, el, 1, quad_rho, quad_theta)
    error = norm(I_exact .- I_num) / norm(I_exact)
    @info "Relative error for h=$(h): $(error)"
end