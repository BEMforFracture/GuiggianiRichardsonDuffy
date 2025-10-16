using Inti, Gmsh
using GLMakie
using StaticArrays
using LinearAlgebra
using LinearAlgebra
using CrackFastBEM

# === INPUTS ===

η = SVector(0.5, 0.5)
ξ = SVector(0.1, 0.6)
el_id = 1

# END === INPUTS ===

Inti.clear_entities!()
gmsh.initialize(String[], true)
msh = Inti.import_mesh("./assets/meshes_template/disks/disk_infinite_media.msh"; dim = 3)
gmsh.finalize()

Γ_msh = view(msh, Inti.Domain(e -> "C" in Inti.labels(e), msh))
Q = Inti.Quadrature(Γ_msh; qorder = 2)

op = Inti.Laplace(; dim = 3)
H = Inti.HyperSingularKernel(op)

w(y) = sqrt(1 - norm(y))

Hw = let w = w, H = H
	(qx, qy) -> H(qx, qy) * w(qy.coords)
end

Inti.singularity_order(::typeof(Hw)) = -3

ℋ = Inti.IntegralOperator(Hw, Q)

el = collect(Inti.elements(Γ_msh))[el_id]
ori = 1

x = el(η)
jac_x = Inti.jacobian(el, η)
nx = Inti._normal(jac_x, ori)
qx = (coords = x, normal = nx)

y = el(ξ)
jac_y = Inti.jacobian(el, ξ)
ny = Inti._normal(jac_y, ori)
qy = (coords = y, normal = ny)
