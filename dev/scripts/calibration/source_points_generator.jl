using StaticArrays
using LinearAlgebra
using Random
using Distributions
using Inti

abstract type AbstractConfig end

struct Quad1N <: AbstractConfig
    el::Inti.LagrangeSquare
    point::SVector{2, Float64}
end

struct Tri1N <: AbstractConfig
    el::Inti.LagrangeTriangle
    point::SVector{2, Float64}
end

struct Quad4N <: AbstractConfig
    el::Inti.LagrangeSquare
    points::Vector{SVector{2, Float64}}
end

struct Tri3N <: AbstractConfig
    el::Inti.LagrangeTriangle
    points::Vector{SVector{2, Float64}}
end

struct Quad9N <: AbstractConfig
    el::Inti.LagrangeSquare
    points::Vector{SVector{2, Float64}}
end

struct Tri6N <: AbstractConfig
    el::Inti.LagrangeTriangle
    points::Vector{SVector{2, Float64}}
end

function build_quad1n_config(el::Inti.LagrangeSquare)
    quad = Inti.GaussLegendre(; order = 1)
    x = quad.nodes[1][1]
    point = SVector(x, x)
    return Quad1N(el, point)
end

function build_tri1n_config(el::Inti.LagrangeTriangle)
    dom = Inti.domain(el)
    quad = Inti.VioreanuRokhlin(; domain = dom, order = 1)
    x = Inti.qcoords(quad)[1]
    return Tri1N(el, x)
end

function build_quad4n_config(el::Inti.LagrangeSquare)
    quad = Inti.GaussLegendre(2)
    _points = Vector{Float64}(undef, 2)
    for i in 1:2
        x = quad.nodes[i][1]
        _points[i] = x
    end
    points = [SVector(x, y) for x in _points for y in _points]
    return Quad4N(el, points)
end

function build_tri3n_config(el::Inti.LagrangeTriangle)
    dom = Inti.domain(el)
    quad = Inti.VioreanuRokhlin(; domain = dom, order = 2)
    points = [SVector(q...) for q in Inti.qcoords(quad)]
    return Tri3N(el, points)
end

function build_quad9n_config(el::Inti.LagrangeSquare)
    quad = Inti.GaussLegendre(3)
    _points = Vector{Float64}(undef, 3)
    for i in 1:3
        x = quad.nodes[i][1]
        _points[i] = x
    end
    points = [SVector(x, y) for x in _points for y in _points]
    return Quad9N(el, points)
end

function build_tri6n_config(el::Inti.LagrangeTriangle)
    dom = Inti.domain(el)
    quad = Inti.VioreanuRokhlin(; domain = dom, order = 4)
    points = [SVector(q...) for q in Inti.qcoords(quad)]
    return Tri6N(el, points)
end