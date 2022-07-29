"""
	_thinqmult(Aqr, tau, B, result)

Multiply `B` by the matrix `Q` obtained from a thin `QR` factorization of `A` using outputs `Aqr` and
`tau` from `LAPACK.geqrf!`. Output is written to `result`.
"""
function _thinqmult!(Aqr::Matrix{F}, tau::Vector{F}, B::Matrix{F}, result::Matrix{F}) where {F <: AbstractFloat}
	m, n = size(Aqr)
	p, q = size(B)
	
	if(n > m)
		throw(DimensionError("first argument must be output from LAPACK.geqrf! with column dimension ("*string(m)*") greater than or equal to row dimension ("*string(n)*")"))
	elseif(n != p)
		throw(DimensionMismatch("row dimension of first argument ("*string(n)*") must equal column dimension of third argument ("*string(p)*")"))
	elseif(size(result) != (m, q))
		throw(DimensionMismatch("dimensions of fourth argument, "*string(size(result))*", must match column dimension of first argument ("*string(m)*") and row dimension of third argument ("*string(q)*")"))
	elseif(length(tau) != n)
		throw(DimensionMismatch("second argument must be output from LAPACK.geqrf! with length ("*string(length(tau))*") equal to row dimension of first argument ("*string(n)*")"))
	end
	
	reflector = Vector{F}(undef, m)		# preallocated memory for a Householder reflector
	
	#= First Householder reflector is applied separately. Since we're multiplying by the thin Q factor,
	we apply only the first n columns of the corresponding Householder transformation. =#
	
	reflector[1] = 1.
	reflector[2:m - n + 1] = Aqr[n + 1:m, n]
	partial = B[n, :]							# equivalent to partial = u[1:n]'*B where u' is the full-length Householder reflector (with leading zeros)
	result[1:m, 1:q] = zeros(m, q); result[1:n, :] = B					# equivalent to result = I[:, 1:n]*B
	result[n:m, :] -= tau[n]*reflector[1:m - n + 1]*partial'	# equivalent to result = I[:, 1:n]*B - tau[n]*u*u[1:n]'*B
	
	# remaining Householder reflectors are applied normally
	
	for i = n - 1:-1:1
		l = m - i + 1
		reflector[2:l] = Aqr[i + 1:m, i]
		partial[1:q] = reflector[1:l]'*result[i:m, :]
		result[i:m, :] -= tau[i]*reflector[1:l]*partial'
	end
end
