using LinearAlgebra

"""
	rqrcp(A, k, p = 0; format = "standard", sketch = gaussianSketchLeft, orthonormal = false)

Compute an approximate factorization `A = R1*R2` where `R1` consists of `k` skeleton
columns from `A`. Choose columns using Businger-Golub QRCP on `sketch(A, k, p)`, where `p` is an
oversampling parameter. If `returnFormat == "minimal"` then only the indices of the skeleton
columns are computed. If `returnFormat == "full"` then `R1`, `R2` are returned along with
the indices of the skeleton columns. If `orthonormal == true` then the columns of `R1`
are orthonormalized. 
"""
function rqrcp(A::Matrix, k::Integer, p::Integer = 0 ;
				format::String = "standard",
				sketch = gaussianSketchLeft,
				orthonormal::Bool = false)
	
	if((k < 0) || (k > min(size(A, 1), size(A, 2))))
		throw(RankError("the target rank must be at least 1 and at most "*string(min(size(A, 1), size(A, 2)))))
	
	elseif(p < 0)
		throw(SketchError("the oversampling parameter must be nonnegative"))
	
	elseif((format != "minimal") && (format != "standard") && (format != "full"))
		throw(ErrorException("options for return format are `minimal`, `standard`, and `full`"))
	end
	
	qrobj = qr(sketch(A, k + p), ColumnNorm())
	perm = qrobj.p[1:k]
	
	if(format == "minimal")
		return perm
	end
	
	R1 = orthonormal ? Matrix(qr(A[:, perm]).Q) : A[:, perm]
	R1p = orthonormal ? R1' : pinv(R1)
	R2 = R1p*A
	
	if(format == "standard")
		return R1, R2
	else
		return R1, R2, perm
	end
end
