using LinearAlgebra

"""
	rsvd(A, k; oversamp = 0, power = 0, minimal = false, sk = GaussianSketch())

Compute an approximate factorization `A = U*diagm(S)*V'` where `U` and `V` consist of `k` orthonormal columns (singular vector estimates )and `S` is a vector of length `k` (singular value estimates) using the algorithm of Halko, Martinsson, and Tropp (2011) with oversampling `oversamp`, `power` steps of power iteration, and sketching specified by `sk`. If `minimal == true` then `U` is not computed and only the singular value estmiates are returned. If  `minimal == false` then `U`, `S`, `V` are returned.
"""
function rsvd(A::Matrix, k::Integer ; 
				oversamp::Integer = 0,
				power::Integer = 0,
				minimal::Bool = false,
				sk::Sketch = GaussianSketch())
	
	if(power < 0)
		throw(ErrorException("power iteration parameter ("*string(power)*") must be nonnegative"))
		
	elseif((k < 0) || (k > min(size(A, 1), size(A, 2))))
		throw(RankError("the target rank ("*string(k)*") must be at least 1 and at most "*string(min(size(A, 1), size(A, 2)))))
	
	elseif(oversamp < 0)
		throw(SketchError("the oversampling parameter ("*string(oversamp)*") must be nonnegative"))
	end
	
	A_sk = sketch(A, k + oversamp, "right", sk)
	Q = Matrix(qr(A_sk).Q)
	
	for i = 1:power
		Q = Matrix(qr(A*(A'*Q)).Q)
	end
	
	svdobj = svd(Q'*A)
	
	if(minimal)
		return svdobj.S[1:k]
	end
	
	U = Q*svdobj.U[:, 1:k]
	V = Matrix(svdobj.V)[:, 1:k]
	
	return RSVD(U, svdobj.S[1:k], V)
end
