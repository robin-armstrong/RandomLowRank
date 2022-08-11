### exceptions specific to the functions of RandomLowRank

# an exception to indicate an invalid target rank
struct RankError <: Exception
	msg::String
end

# an exception to indicate problems in computing a sketch
struct SketchError <: Exception
	msg::String
end
