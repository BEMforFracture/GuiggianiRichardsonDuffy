using Inti
using StaticArrays
using GuiggianiRichardsonDuffy

## first step : build a test case element
y1 = SVector(0.0, 0.0, 0.0)
y2 = SVector(1.0, 0.0, 0.0)
y3 = SVector(0.0, 1.0, 0.0)
element = Inti.LagrangeTriangle(y1, y2, y3)

## second step : kernel
op = Inti.Elastostatic(λ = 1.0, μ = 1.0; dim = 3)

K = Inti.DoubleLayerKernel(op)
# K = Inti.AdjointDoubleLayerKernel(op)
# K = Inti.HyperSingularKernel(op)

## third step : auxiliary parameters

ori = 1
quad_rho = Inti.GaussLegendre(5)
quad_theta = Inti.GaussLegendre(10)

regular_quad_1D = Inti.GaussLegendre(2)

domain = Inti.reference_domain(element)
regular_quad =  Inti.VioreanuRokhlin(; domain = domain, order = 2)

x, w = regular_quad()

id_x = 1

L = Inti.lagrange_basis(regular_quad)

x̂ = x[id_x]
û = ξ -> L(ξ)[id_x]

method_rich = FullRichardsonExpansion()
method_diff = AutoDiffExpansion()
method_semi_rich = SemiRichardsonExpansion()

## fourth step : compute the integrals
I_rich = GuiggianiRichardsonDuffy.guiggiani_singular_integral(K, û, x̂, element, ori, quad_rho, quad_theta, method_rich)
I_diff = GuiggianiRichardsonDuffy.guiggiani_singular_integral(K, û, x̂, element, ori, quad_rho, quad_theta, method_diff)
I_semi_rich = GuiggianiRichardsonDuffy.guiggiani_singular_integral(K, û, x̂, element, ori, quad_rho, quad_theta, method_semi_rich)
I_ref = Inti.guiggiani_singular_integral(K, û, x̂, element, ori, quad_rho, quad_theta)
## play with splitted kernel

SK = GuiggianiRichardsonDuffy.SplitKernel(K)

id_y = 2
ŷ = x[id_y]

nx = Inti.normal(element, x̂)
ny = Inti.normal(element, ŷ)

qx = (coords = element(x̂), normal = nx)
qy = (coords = element(ŷ), normal = ny)

## play with laurent coefficients

L_rich = GuiggianiRichardsonDuffy.laurents_coeffs(K, element, ori, û, x̂, method_rich)
L_diff = GuiggianiRichardsonDuffy.laurents_coeffs(K, element, ori, û, x̂, method_diff)
L_semi_rich = GuiggianiRichardsonDuffy.laurents_coeffs(K, element, ori, û, x̂, method_semi_rich)