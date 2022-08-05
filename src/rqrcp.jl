using LinearAlgebra

"""
	rqrcp(A, k, [p = 0, qrmat, jpvt, tau, basis]; orthobasis = false)

Select `k` skeleton columns of a real matrix  `A` using QRCP on a randomized
sketch with `k + p` columns. If `orthobasis == true` then also compute an 
orthonormal basis for the span of the skeleton columns. Optional arguments 
`qrmat`, `jpvt`, and `tau` provide preallocated memory for `LAPACK.geqp3!`.
Optional argument `basis` provides preallocated memory to store the orthonormal
basis vectors.
"""
function rqrcp(A::Matrix{F}, k::Integer, p::Integer = 0,
				sketch::Matrix{F} = Matrix{F}(undef, k + p, size(A, 2)),
				jpvt::Vector{Itg} = Vector{Int64}(undef, size(A, 2)),
				tau::Vector{F} = Vector{F}(undef, k + p),
				basis::Union{Matrix{F}, Nothing} = Nothing ;
				orthobasis::Bool = false
				) where {F <: AbstractFloat, Itg <: Integer}
	
	if(k < 1)
		throw(DimensionError("second argument must be a positive integer"))
	elseif(p < 0)
		throw(DimensionError("oversampling parameter must be a nonnegative integer"))
	elseif(k + p > size(A, 2))
		throw(DimensionError("row dimension of sketch cannot exceed row dimension of first argument"))
	elseif(size(sketch) != (k + p, size(A, 2)))
		throw(DimensionMismatch("preallocated sketching matrix must have dimensions ("*string(k + p)*", "*string(size(A, 2))*")"))
	elseif(length(jpvt) != size(A, 2))
		throw(DimensionMismatch("preallocated vector of pivots must have length "*string(size(A, 2))))
	elseif(length(tau) != k + p)
		throw(DimensionMismatch("preallocated vector of Householder scalars must have length "*string(k + p)))
	elseif(orthobasis)
		if((basis != Nothing) && (size(basis) != (size(A, 1), k)))
			throw DimensionMismatch("preallocated basis matrix must have dimensions ("*string(size(A, 1))*", "*string(k)*")")
		end
	end
	
	sketch[:, :] = randn(k + p, size(A, 1))*A
	LAPACK.geqp3!(sketch, jpvt, tau)
	
	if(orthobasis)
		if(basis == Nothing)
			basis = Matrix{F}(I(size(A, 1)))[1:k]
		end
	else
		return jpvt[1:k]
	end
end
