module TriPMatrices

# package code goes here
include("tri_per.jl")
module Lapack
include("lapack.jl")
end
include("lu.jl")


end # module
