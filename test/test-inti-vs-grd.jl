using Inti
using Gmsh
using LinearAlgebra
using SparseArrays
using GuiggianiRichardsonDuffy

a = 1.0
b = 0.5

meshsize = 0.025
qorder = 2

# Create a simple ellipse

function create_ellipse(a, b, meshsize)
    gmsh.initialize()
    gmsh.model.add("ellipse")
    lc = meshsize
    el = gmsh.model.occ.addEllipse(0.0, 0.0, 0.0, a, b, -1)
    curve_loop = gmsh.model.occ.addCurveLoop([el])
    crack = gmsh.model.occ.addPlaneSurface([el], -1)
    gmsh.model.occ.synchronize()
    pg_crack = gmsh.model.addPhysicalGroup(2, [crack], -1, "C")
    pg_front = gmsh.model.addPhysicalGroup(1, [el], -1, "F")
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)
    mesh = Inti.import_mesh(; dim = 3)
    gmsh.finalize()
    return mesh
end

mesh = create_ellipse(a, b, meshsize)

crack_mesh = view(mesh, Inti.Domain(e -> "C" in Inti.labels(e), mesh))

Q_crack = Inti.Quadrature(crack_mesh; qorder = qorder)

op = Inti.Elastostatic(; μ = 1.0, λ = 1.0, dim = 3)
H = Inti.HyperSingularKernel(op)
ℋ = Inti.IntegralOperator(H, Q_crack, Q_crack)

GC.gc()
δK = @timed adaptive_correction(ℋ; method = AutoDiffExpansion(), maxdist = meshsize)
@info "Adaptive correction time: $(δK.time) seconds"
GC.gc()
δK_inti = @timed Inti.adaptive_correction(ℋ; maxdist = meshsize)
@info "Adaptive correction time (Inti): $(δK_inti.time) seconds"