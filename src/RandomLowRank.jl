module RandomLowRank

export rqrcp, rgks, rsvd, rheigen, noSketch, gaussianSketch, RSVD, SkeletalDecomp, OrthoSkeletalDecomp, RankError, SketchError

include("exceptions.jl")
include("sketches.jl")
include("returnstructs.jl")

include("rqrcp.jl")
include("rgks.jl")
include("rsvd.jl")
include("rheigen.jl")

end
