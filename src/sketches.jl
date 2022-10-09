using Hadamard

"""
An abstract type whose subtypes represent different procedures for matrix sketching. For any concrete type `ConcreteSketch <: Sketch` the method

	sketch(A::Matrix, l::Integer, side::String, sk::ConcreteSketch)
	
must be implemented. This method returns the sketch of `A` according to the procedure that `ConcreteSketch` specifies, in such a way that the column dimension is preserved (for `side == "right"`) or the row dimension is preserved for `side == "left"`). The non-preserved dimension of the returned matrix must be at least `l`.
"""
abstract type Sketch
end

"""
A "do-nothing" sketch to allow for deterministic use of sketching-based algorithms.
"""
struct NoSketch <: Sketch
end

function sketch(A::Matrix, l::Integer, side::String, sk::NoSketch)
	return deepcopy(A)
end

"""
Gaussian sketching, wherein the input matrix is multiplied by a standard Gaussian matrix.
"""
struct GaussianSketch <: Sketch
end

function sketch(A::Matrix, l::Integer, side::String, sk::GaussianSketch)
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

"""
Matrix sketching via a subsampled random Hadamard transform.
"""
struct SRHTSketch <: Sketch
end

function sketch(A::Matrix, l::Integer, side::String, sk::SRHTSketch)
	if(l < 1)
	    throw(SketchError("target dimension of sketch must be positive"))
	elseif(side == "left")
		if(l > size(A, 1))
	        throw(SketchError("column dimension of sketch ("*string(l)*") cannot exceed column dimension of matrix ("*string(size(A, 1))*")"))
	    end
		
		# padding with zeros to make the matrix conformal with a Walsh-Hadamard transform
	    
	    hadamardDim = 1
	    while(hadamardDim < size(A, 1))
	    	hadamardDim *= 2
	    end
	    
	    presketch = [A; zeros(hadamardDim - size(A, 1), size(A, 2))]
	    
	    # applying random row scaling
	    
	    for i = 1:hadamardDim
	    	presketch[i, :] *= rand() < .5 ? hadamardDim : -hadamardDim
	    end
	    
	    presketch = fwht(presketch, 1)	# Walsh-Hadamard transform on the columns
	    rowindices = broadcast(x -> ceil(Int64, hadamardDim*x), rand(l))
	    
	    return presketch[rowindices, :]/sqrt(l)
	
	elseif(side == "right")
		if(l > size(A, 2))
	        throw(SketchError("row dimension of sketch ("*string(l)*") cannot exceed row dimension of matrix ("*string(size(A, 2))*")"))
	    end
	    
	    # padding with zeros to make the matrix conformal with a Walsh-Hadamard transform
	    
	    hadamardDim = 1
	    while(hadamardDim < size(A, 2))
	    	hadamardDim *= 2
	    end
	    
	    presketch = [A zeros(size(A, 1), hadamardDim - size(A, 2))]
	    
	    # applying random column scaling
	    
	    for i = 1:hadamardDim
	    	presketch[:, i] *= rand() < .5 ? hadamardDim : -hadamardDim
	    end
	    
	    presketch = fwht(presketch, 2)	# Walsh-Hadamard transform on the rows
	    colindices = broadcast(x -> ceil(Int64, hadamardDim*x), rand(l))
	    
	    return presketch[:, colindices]/sqrt(l)
	    
	else
	    throw(SketchError("unsupported sketching specifier '"*side*"'"))
	end
end
