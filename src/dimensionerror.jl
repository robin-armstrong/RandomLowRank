"""
	DimensionError(msg)

Error type to indicate inappropriate matrix dimensions in cases not
covered by DimensionMismatch.
"""
struct DimensionError <: Exception
	msg::String
end
