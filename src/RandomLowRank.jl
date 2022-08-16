module RandomLowRank

# lowrank methods
export rqrcp, rgks, rsvd, rheigen

# structs and methods related to sketching
export sketch, Sketch, NoSketch, GaussianSketch

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
