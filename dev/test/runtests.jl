using Test
import GuiggianiRichardsonDuffy as GRD

@info "Running GuiggianiRichardsonDuffy.jl development tests"

GRD.@suppress_output include("smoketests.jl")
