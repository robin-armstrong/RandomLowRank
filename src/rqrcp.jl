using LinearAlgebra

"""
	rqrcp(A, k, p = 0; format = "full", sk = gaussianSketch(), orthonormal = false)

Compute an approximate factorization `A = C*B` where `C` consists of `k` skeleton
columns from `A`. Choose columns using Businger-Golub QRCP on a sketch of `A` computed
according to `sk`, using oversampling `p`. If `returnFormat == "minimal"` then only the
indices of the skeleton columns are computed. If `returnFormat == "full"` then `C`, `B`
are returned along with the indices of the skeleton columns. If `orthonormal == true` 
then the columns of `C` are orthonormalized. 
"""
function rqrcp(A::Matrix, k::Integer, p::Integer = 0 ;
				format::String = "full",
				sk::Sketch = gaussianSketch(),
				orthonormal::Bool = false)
	
	if((k < 0) || (k > min(size(A, 1), size(A, 2))))
		throw(RankError("the target rank ("*string(k)*") must be at least 1 and at most "*string(min(size(A, 1), size(A, 2)))))
	
	elseif(p < 0)
		throw(SketchError("the oversampling parameter (*string(p)*) must be nonnegative"))
	
	elseif((format != "minimal") && (format != "full"))
		throw(ErrorException("options for return format are `minimal` and `full`"))
	end
	
	A_sk = sketch(A, k + p, "left", sk)
	qrobj = qr(A_sk, ColumnNorm())
	perm = qrobj.p[1:k]
	
	if(format == "minimal")
		return perm
	end
	
	C = orthonormal ? Matrix(qr(A[:, perm]).Q) : A[:, perm]
	Cp = orthonormal ? C' : pinv(C)
	B = Cp*A
	
	return orthonormal ? OrthoSkeletalDecomp(perm, C, B) : SkeletalDecomp(perm, C, B)
end
