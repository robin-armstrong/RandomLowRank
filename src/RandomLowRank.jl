module RandomLowRank

export rqrcp, rgks, rsvd, noSketch, gaussianSketch, RSVD, SkeletalDecomp, OrthoSkeletalDecomp, RankError, SketchError

include("exceptions.jl")
include("sketches.jl")
include("returnstructs.jl")

include("rqrcp.jl")
include("rgks.jl")
include("rsvd.jl")

end
