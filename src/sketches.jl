"""
Functions to compute randomized sketches of matrices. Each has the signature
`mySketch(A::Matrix, l::Integer)`, where`A` is the matrix being sketched and
`l` 
"""
function noSketch(A::Matrix, l::Integer)
	return deepcopy(A)
end

function gaussianSketchLeft(A::Matrix, l::Integer)
	l = min(l, size(A, 1))
	return randn(l, size(A, 1))*A
end

function gaussianSketchRight(A::Matrix, l::Integer)
	l = min(l, size(A, 2))
	return A*randn(size(A, 2), l)
end
