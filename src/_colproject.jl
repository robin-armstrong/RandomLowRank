using LinearAlgebra
include("dimensionerror.jl")

"""
    _colproject!(A, B)

Orthogonally project the columns of `A` into the column space of `B`. Modifies `A` in place.
"""
function _colproject!(A::Matrix{F}, B::Matrix{F}) where {F <: AbstractFloat}
	if(size(A, 1) != size(B, 1))
		throw(DimensionMismatch("column dimension of A ("*string(size(A, 1))*") must match column dimension of B ("*string(size(B, 2))*")"))
	elseif(size(B, 2) > size(B, 1))
        throw(DimensionError("row dimension of B ("*string(size(B, 2))*") cannot be larger than column dimension ("*string(size(B, 1))*")"))
	end
	
	tau = Vector{Float64}(undef, size(B, 2))    # parameters for Householder reflectors
	qrobj = deepcopy(B)
	LAPACK.geqrf!(qrobj, tau)       # compute the QR factorization
end
