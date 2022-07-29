using RandomLowRank
using LinearAlgebra
using Test

@testset "RandomLowRank._thinqmult! tests" begin
	# testing with a tall and thin Q
	
	A = randn(10, 3)
	qrobj = deepcopy(A)
	tau = zeros(3)
	LAPACK.geqrf!(qrobj, tau)

	B = randn(3, 10)
	QB = zeros(10, 10)
	RandomLowRank._thinqmult!(qrobj, tau, B, QB)

	Q = Matrix(qr(A).Q)
	@test norm(QB - Q*B) < 1e-12
	
	# testing with a square Q
	
	A = randn(10, 10)
	qrobj = deepcopy(A)
	tau = zeros(10)
	LAPACK.geqrf!(qrobj, tau)

	B = randn(10, 10)
	QB = zeros(10, 10)
	RandomLowRank._thinqmult!(qrobj, tau, B, QB)

	Q = Matrix(qr(A).Q)
	@test norm(QB - Q*B) < 1e-12
end
