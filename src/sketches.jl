#=
Functions to compute randomized sketches of matrices. Each has the signature
`mySketch(A::Matrix, l::Integer, side::String)`, where`A` is the matrix being
sketched and `l` indicates the target dimension of the sketch. If `side == "left"`
then the returned matrix has dimensions `(l, size(A, 2))`, and if `side == "right"`
then the returned matrix has dimensions `(size(A, 1), l)`.
=#

# a "do-nothing" sketch to allow for deterministic use of sketching-based algorithms.
function noSketch(A::Matrix, l::Integer, side::String)
	return deepcopy(A)
end

function gaussianSketch(A::Matrix, l::Integer, side::String)
	if(l < 1)
	    throw(SketchError("target dimension of sketch must be positive"))
	elseif(side == "left")
	    if(l > size(A, 1))
	        throw(SketchError("column dimension of sketch ("*string(l)*") cannot exceed column dimension of matrix ("*string(size(A, 1))*")"))
	    end
	    
	    return randn(l, size(A, 1))*A
	elseif(side == "right")
	    if(l > size(A, 2))
	        throw(SketchError("row dimension of sketch ("*string(l)*") cannot exceed row dimension of matrix ("*string(size(A, 2))*")"))
	    end
	    
	    return A*randn(size(A, 2), l)
	else
	    throw(SketchError("unsupported sketching specifier '"*side*"'"))
	end
end

