"""
Randomized low-rank algorithms for numerical linear algebra. Provides functions for computing
approximate eigenvalue, singular value, and interpolative decompositions using randomized
sketching.
"""
module RandomLowRank

# lowrank methods
export rqrcp, rgks, rsvd, rheigen

# structs and methods related to sketching
export sketch, Sketch, NoSketch, GaussianSketch, HadamardSketch, CWSketch

# structs returned by lowrank methods
export RSVD, RHEigen, SkeletalDecomp, OrthoSkeletalDecomp

# custom exceptions
export RankError, SketchError

include("exceptions.jl")
include("sketches.jl")
include("returnstructs.jl")

include("rqrcp.jl")
include("rgks.jl")
include("rsvd.jl")
include("rheigen.jl")

end
