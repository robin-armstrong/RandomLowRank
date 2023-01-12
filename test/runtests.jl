using RandomLowRank
using LinearAlgebra
using Random
using Test

rng = MersenneTwister(1)	# setting an explicit seed for reproducibility

# function to print debug messages for failed tests
function showInfo(msg, testResult)
	if(typeof(testResult) != Test.Pass)
		@info msg
	end
end

# constructing test matrices

largeDim = 200
smallDim = 100
numericalRank = 10
residual = 1e-8

U_true = Matrix(qr(randn(rng, largeDim, smallDim)).Q)
V_true = Matrix(qr(randn(rng, smallDim, smallDim)).Q)
S_true = ones(smallDim)

rho = residual^(1/(numericalRank - 1))

for i = 1:numericalRank - 1
	S_true[i + 1:end] *= rho
end

A_tall = U_true*diagm(S_true)*V_true'
A_wide = Matrix(A_tall')

@testset "sketching tests" begin
	M = randn(rng, 100, 100)

	M_sk = sketch(M, 50, "left", GaussianSketch(), rng)
	@test size(M_sk) == (50, 100)

	M_sk = sketch(M, 50, "right", GaussianSketch(), rng)
	@test size(M_sk) == (100, 50)

	M_sk = sketch(M, 50, "left", NoSketch(), rng)
	@test M_sk == M

	M_sk = sketch(M, 50, "left", HadamardSketch(), rng)
	@test size(M_sk) == (50, 100)

	M_sk = sketch(M, 50, "right", HadamardSketch(), rng)
	@test size(M_sk) == (100, 50)

	M_sk = sketch(M, 50, "left", CWSketch(), rng)
	@test size(M_sk) == (50, 100)

	M_sk = sketch(M, 50, "right", CWSketch(), rng)
	@test size(M_sk) == (100, 50)
end

@testset "rng tests" begin
	A = randn(rng, 50, 50)
	k = 25
	
	for s in [GaussianSketch, HadamardSketch, CWSketch]
		params = "parameters are sk = "*string(s())
		
		U1, S1, Vt1 = rsvd(MersenneTwister(2), A, k, sk = s())
		U2, S2, Vt2 = rsvd(MersenneTwister(2), A, k, sk = s())
		U3, S3, Vt3 = rsvd(A, k, sk = s())
		
		showInfo(params, @test U1*diagm(S1)*Vt1 == U2*diagm(S2)*Vt2)
		showInfo(params, @test U1*diagm(S1)*Vt1 != U3*diagm(S3)*Vt3)
		
		_, C1, R1 = rgks(MersenneTwister(2), A, k, sk = s())
		_, C2, R2 = rgks(MersenneTwister(2), A, k, sk = s())
		_, C3, R3 = rgks(A, k)
		
		showInfo(params, @test C1*R1 == C2*R2)
		showInfo(params, @test C1*R1 != C3*R3)
		
		_, C1, R1 = rqrcp(MersenneTwister(2), A, k, sk = s())
		_, C2, R2 = rqrcp(MersenneTwister(2), A, k, sk = s())
		_, C3, R3 = rqrcp(A, k, sk = s())
		
		showInfo(params, @test C1*R1 == C2*R2)
		showInfo(params, @test C1*R1 != C3*R3)
		
		lambda1, V1 = rheigen(MersenneTwister(2), A + A', k, sk = s())
		lambda2, V2 = rheigen(MersenneTwister(2), A + A', k, sk = s())
		lambda3, V3 = rheigen(A + A', k, sk = s())
		
		showInfo(params, @test V1*diagm(lambda1)*V1' == V2*diagm(lambda2)*V2')
		showInfo(params, @test V1*diagm(lambda1)*V1' != V3*diagm(lambda3)*V3')
	end
end

@testset "rqrcp tests" begin
	mats = [A_tall, A_wide]
	matnames = ["A_tall", "A_wide"]
	
	for i = 1:2
		A = mats[i]
		A_name = matnames[i]
		
		for s in [NoSketch, GaussianSketch, HadamardSketch, CWSketch]
			for p in [0, 5, smallDim - numericalRank]
				params = "parameters are A = "*A_name*", s = "*string(s)*", p = "*string(p)
				
				perm = rqrcp(rng, A, numericalRank, oversamp = p, minimal = true, sk = s())
				Q = Matrix(qr(A[:, perm]).Q)
				err = opnorm(A - Q*Q'*A)/residual
				
				if(err > 50)
					@warn "rqrcp tests: relative error from minimal call exceeds 50"
					@info params
				end
				
				showInfo(params, @test length(perm) == numericalRank)
				
				perm, C, B = rqrcp(rng, A, numericalRank, oversamp = p, sk = s())
				err = opnorm(A - C*B)/residual
				
				if(err > 50)
					@warn "rqrcp tests: relative error from full call exceeds 50"
					@info params
				end
				
				showInfo(params, @test size(C) == (size(A, 1), numericalRank))
				showInfo(params, @test size(B) == (numericalRank, size(A, 2)))
				showInfo(params, @test length(perm) == numericalRank)
				
				# testing orthonormalization
				_, Q, B = rqrcp(rng, A, numericalRank, oversamp = p, sk = s(), orthonormal = true)
				showInfo(params, @test size(Q) == (size(A, 1), numericalRank))
				showInfo(params, @test size(B) == (numericalRank, size(A, 2)))
				showInfo(params, @test opnorm(Q'*Q - I(numericalRank)) < 1e-10)
			end
		end
	end
end

@testset "rgks tests" begin
	mats = [A_tall, A_wide]
	matnames = ["A_tall", "A_wide"]
	
	for i = 1:2
		A = mats[i]
		A_name = matnames[i]
		
		for s in [NoSketch, GaussianSketch, HadamardSketch, CWSketch]
			for p in [0, 5, smallDim - numericalRank]
				params = "parameters are A = "*A_name*", s = "*string(s)*", p = "*string(p)
				
				perm = rgks(rng, A, numericalRank, oversamp = p, minimal = true, sk = s())
				Q = Matrix(qr(A[:, perm]).Q)
				err = opnorm(A - Q*Q'*A)/residual
				
				if(err > 50)
					@warn "rgks tests: relative error from minimal call exceeds 50"
					@info params
				end
				
				showInfo(params, @test length(perm) == numericalRank)
				
				perm, C, B = rgks(rng, A, numericalRank, oversamp = p, sk = s())
				err = opnorm(A - C*B)/residual
				
				if(err > 50)
					@warn "rgks tests: relative error from full call exceeds 50"
					@info params
				end
				
				showInfo(params, @test size(C) == (size(A, 1), numericalRank))
				showInfo(params, @test size(B) == (numericalRank, size(A, 2)))
				showInfo(params, @test length(perm) == numericalRank)
				
				# testing orthonormalization
				_, Q, B = rgks(rng, A, numericalRank, oversamp = p, sk = s(), orthonormal = true)
				showInfo(params, @test size(Q) == (size(A, 1), numericalRank))
				showInfo(params, @test size(B) == (numericalRank, size(A, 2)))
				showInfo(params, @test opnorm(Q'*Q - I(numericalRank)) < 1e-10)
			end
		end
	end
end

@testset "rsvd tests" begin
	mats = [A_tall, A_wide]
	matnames = ["A_tall", "A_wide"]
	
	for i = 1:2
		A = mats[i]
		A_name = matnames[i]
		
		for s in [NoSketch, GaussianSketch, HadamardSketch, CWSketch]
			for p in [0, 5, smallDim - numericalRank]
				for q in [0, 2, 4]
					params = "parameters are A = "*A_name*", s = "*string(s)*", p = "*string(p)*", power = "*string(q)
					
					S = rsvd(rng, A, numericalRank, oversamp = p, power = q, sk = s(), minimal = true)
					showInfo(params, @test length(S) == numericalRank)
					
					errvect = broadcast(i -> (S[i] - S_true[i])^2/S_true[i]^2, 1:numericalRank)
					err = sqrt(sum(errvect)/numericalRank)
					
					if(err > 50)
						@warn "rsvd tests: minimal format singular value relative error exceeds 50"
						@info params
					end
					
					U, S, Vt = rsvd(rng, A, numericalRank, oversamp = p, power = q, sk = s())
					showInfo(params, @test size(U) == (size(A, 1), numericalRank))
					showInfo(params, @test size(Vt) == (numericalRank, size(A, 2)))
					showInfo(params, @test length(S) == numericalRank)
					
					singvals_errvect = broadcast(i -> (S[i] - S_true[i])^2/S_true[i]^2, 1:numericalRank)
					singvals_err = sqrt(sum(singvals_errvect)/numericalRank)
					
					if(singvals_err > 50)
						@warn "rsvd tests: full format singular value relative error exceeds 50"
						@info params
					end
					
					err = opnorm(A - U*diagm(S)*Vt)/residual
					
					if(err > 50)
						@warn "rsvd tests: full format matrix estimation relative error exceeds 50"
						@info params
					end
				end
			end
		end
	end
end

@testset "rheigen tests" begin
	V_true = Matrix(qr(randn(rng, largeDim, largeDim)).Q)
	lambda_true = residual*ones(largeDim)
	lambda_true[1:5] = [100., -50., 30., -15., 5.]
	A = V_true*diagm(lambda_true)*V_true'
	numValsToTest = 5
	
	for s in [NoSketch, GaussianSketch, HadamardSketch, CWSketch]
		for p in [0, 5, largeDim - numValsToTest]
			for q in [0, 2, 4]
				params = "parameters are s = "*string(s)*", p = "*string(p)*", power = "*string(q)
				
				lambda = rheigen(rng, A, numValsToTest, oversamp = p, power = q, sk = s(), minimal = true)
				showInfo(params, @test length(lambda) == numValsToTest)
				
				errvect = broadcast(i -> (lambda[i] - lambda_true[i])^2/lambda_true[i]^2, 1:numValsToTest)
				err = sqrt(sum(errvect)/numValsToTest)
				
				if(err > 50)
					@warn "rheigen tests: minimal format eigenvalue relative error exceeds 50"
					@info params
				end
				
				lambda, V = rheigen(rng, A, numValsToTest, oversamp = p, power = q, sk = s())
				showInfo(params, @test length(lambda) == numValsToTest)
				showInfo(params, @test size(V) == (largeDim, numValsToTest))
				
				values_errvect = broadcast(i -> (lambda[i] - lambda_true[i])^2/lambda_true[i]^2, 1:numValsToTest)
				values_err = sqrt(sum(values_errvect)/numValsToTest)
				
				if(values_err > 50)
					@warn "rheigen tests: full format eigenvalue relative error exceeds 50"
					@info params
				end
				
				err = opnorm(A - V*diagm(lambda)*V')/residual
				
				if(err > 50)
					@warn "rheigen tests: full format matrix estimation relative error exceeds 50"
					@info params
				end
			end
		end
	end
end
