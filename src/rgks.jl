using LinearAlgebra

"""
	rgks(A, k, p = 0; minimal = false, sk = GaussianSketch(), orthonormal = false)

Compute an approximate factorization `A = C*B` where `C` consists of `k` skeleton columns from `A`. Choose columns using the procedure of Golub, Klemma, and Stewart (1976) on a sketch of `A` computed according to `sk`, using oversampling `p`. If `minimal == true` then only the indices of the skeleton columns are computed. If `minimal == false` then `C`, `B` are returned along with the indices of the skeleton columns. If `orthonormal == true` then the columns of `C` are orthonormalized. 
"""
function rgks(A::Matrix, k::Integer, p::Integer = 0 ;
				minimal::Bool = false,
				sk::Sketch = GaussianSketch(),
				orthonormal::Bool = false)
	
	if((k < 0) || (k > min(size(A, 1), size(A, 2))))
		throw(RankError("the target rank ("*string(k)*") must be at least 1 and at most "*string(min(size(A, 1), size(A, 2)))))
	
	elseif(p < 0)
		throw(SketchError("the oversampling parameter ("*string(p)*") must be nonnegative"))
	end
	
	A_sk = sketch(A, k + p, "right", sk)
	Q = Matrix(qr(A_sk).Q)
	V = Matrix(svd(Q'*A).V)[:, 1:k]
	qrobj = qr(V', ColumnNorm())
	perm = qrobj.p[1:k]
	
	if(minimal)
		return perm
	end
	
	C = orthonormal ? Matrix(qr(A[:, perm]).Q) : A[:, perm]
	Cp = orthonormal ? C' : pinv(C)
	B = Cp*A
	
	return orthonormal ? OrthoSkeletalDecomp(perm, C, B) : SkeletalDecomp(perm, C, B)
end
