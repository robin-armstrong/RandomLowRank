using LinearAlgebra

"""
	rheigen(A, k, p = 0; power = 0, format = "full", sk = gaussianSketch(), tol = 1e-12)

Compute an approximate factorization `A = V*diagm(lambda)*V'` for the Hermitian matrix `A`, where
`V` consists of `k` orthonormal columns (eigenvector estimates) and `lambda` is a vector of
length `k` (eigenvalue estimates) using the algorithm of Halko, Martinsson, and Tropp (2011) 
with oversampling `p`, `power` steps of power iteration, and sketching specified by `sk`.
Estimates correspond to the `k` eigenvalues of `A` which are largest in magnitude. If 
`format == "minimal"` then only the eigenvalue estimates are computed. If `format == "full"` then
`lambda, V` are returned. An error will be thrown if `A` differs from `Hermitian(A)` by more than
`tol` in the Frobenius norm.
"""
function rheigen(A::Matrix, k::Integer, p::Integer = 0 ;
					power::Integer = 0,
					format::String = "full",
					sk::Sketch = gaussianSketch(),
					tol = 1e-12)
	
	if(size(A, 1) != size(A, 2))
		throw(ErrorException("matrix is not square: dimensions are "*string(size(A))))
	end
	
	A_herm = Hermitian(A)
	
	if(norm(A - A_herm) > tol)
		throw(ErrorException("matrix must be Hermitian"))
	
	elseif(power < 0)
		throw(ErrorException("power iteration parameter ("*string(power)*") must be nonnegative"))
		
	elseif((k < 0) || (k > min(size(A, 1), size(A, 2))))
		throw(RankError("the target rank ("*string(k)*") must be at least 1 and at most "*string(min(size(A, 1), size(A, 2)))))
	
	elseif(p < 0)
		throw(SketchError("the oversampling parameter ("*string(p)*") must be nonnegative"))
	
	elseif((format != "minimal") && (format != "full"))
		throw(ErrorException("options for return format are `minimal` and `full`"))
	end
	
	A_sk = sketch(A, k + p, "right", sk)
	Q = Matrix(qr(A_sk).Q)
	
	for i = 1:power
		Q = Matrix(qr(A*Q).Q)
	end
	
	eigenobj = eigen(Hermitian(Q'*(A*Q)))
	
	if(format == "minimal")
		return eigenobj.values[1:k]
	end
	
	perm = sortperm(eigenobj.values, by = t -> -abs(t))
	lambda = eigenobj.values[perm[1:k]]
	V = Q*eigenobj.vectors[:, perm[1:k]]
	perm = sortperm(lambda)
	
	return RHEigen(lambda[perm], V[:, perm])
end
