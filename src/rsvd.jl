using LinearAlgebra

"""
	rsvd(A, k, p = 0; power = 0, format = "full", sketch = gaussianSketch)

Compute an approximate factorization `A = U*diagm(S)*V'` where `U` and `V` consist
of `k` orthonormal columns (singular vector estimates )and `S` is a vector of length `k`
(singular value estimates) using the algorithm of Halko, Martinsson, and Tropp (2011) 
with oversampling `p`, `power` steps of power iteration, and sketching specified by `sketch`.
If `format == "minimal"` then `U` is not computed and only the singular value estmiates are 
returned. If  `format == "full"` then `U`, `S`, `V` are returned.
"""
function rsvd(A::Matrix, k::Integer, p::Integer = 0 ;
				power::Integer = 0,
				format::String = "full",
				sketch = gaussianSketch)
	
	if(power < 0)
		throw(ErrorException("power iteration parameter ("*string(power)*") must be nonnegative"))
		
	elseif((k < 0) || (k > min(size(A, 1), size(A, 2))))
		throw(RankError("the target rank ("*string(k)*") must be at least 1 and at most "*string(min(size(A, 1), size(A, 2)))))
	
	elseif(p < 0)
		throw(SketchError("the oversampling parameter ("*string(p)*") must be nonnegative"))
	
	elseif((format != "minimal") && (format != "full"))
		throw(ErrorException("options for return format are `minimal` and `full`"))
	end
	
	sk = sketch(A, k + p, "right")
	Q = Matrix(qr(sk).Q)
	
	for i = 1:power
		Q = Matrix(qr(A*(A'*Q)).Q)
	end
	
	svdobj = svd(Q'*A)
	
	if(format == "minimal")
		return svdobj.S[1:k]
	end
	
	U = Q*svdobj.U[:, 1:k]
	V = Matrix(svdobj.V)[:, 1:k]
	
	return RSVD(U, svdobj.S[1:k], V)
end
