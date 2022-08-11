using RandomLowRank
using LinearAlgebra
using Test

# function to print debug messages for failed test
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

U = Matrix(qr(randn(largeDim, smallDim)).Q)
V = Matrix(qr(randn(smallDim, smallDim)).Q)
S = ones(smallDim)

for i = 1:numericalRank
	S[i] = numericalRank - i + 1.
end

for i = numericalRank + 1:smallDim
	S[i] = residual
end

A_tall = U*diagm(S)*V'
A_wide = Matrix(A_tall')

@testset "rqrcp tests" begin
	mats = [A_tall, A_wide]
	matnames = ["A_tall", "A_wide"]
	
	for i = 1:2
		A = mats[i]
		A_name = matnames[i]
		
		for sk in [noSketch, gaussianSketchLeft, gaussianSketchRight]
			for p in [0, 5, smallDim - numericalRank]
				params = "parameters are A = "*A_name*", sk = "*string(sk)*", p = "*string(p)
				
				R1, R2 = rqrcp(A, numericalRank, p, sketch = sk)
				err = opnorm(A - R1*R2)*1e8
				
				if(err > 1000)
					@warn("rqrcp tests: relative error from standard call exceeds 1000")
					@info params
				end
				
				showInfo(params, @test size(R1) == (size(A, 1), numericalRank))
				showInfo(params, @test size(R2) == (numericalRank, size(A, 2)))
				
				perm = rqrcp(A, numericalRank, p, format = "minimal", sketch = sk)
				Q = Matrix(qr(A[:, perm]).Q)
				err = opnorm(A - Q*Q'*A)*1e8
				
				if(err > 1000)
					@warn("rqrcp tests: relative error from minimal call exceeds 1000")
					@info params
				end
				
				showInfo(params, @test length(perm) == numericalRank)
				
				R1, R2, perm = rqrcp(A, numericalRank, p, format = "full", sketch = sk)
				err = opnorm(A - R1*R2)*1e8
				
				if(err > 1000)
					@warn("rqrcp tests: relative error from full call exceeds 1000")
					@info params
				end
				
				showInfo(params, @test size(R1) == (size(A, 1), numericalRank))
				showInfo(params, @test size(R2) == (numericalRank, size(A, 2)))
				showInfo(params, @test length(perm) == numericalRank)
				
				# testing orthonormalization
				R1, R2 = rqrcp(A, numericalRank, p, sketch = sk, orthonormal = true)
				showInfo(params, @test size(R1) == (size(A, 1), numericalRank))
				showInfo(params, @test size(R2) == (numericalRank, size(A, 2)))
				showInfo(params, @test opnorm(R1'*R1 - I(numericalRank)) < 1e-10)
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
		
		for sk in [noSketch, gaussianSketchLeft, gaussianSketchRight]
			for p in [0, 5, smallDim - numericalRank]
				params = "parameters are A = "*A_name*", sk = "*string(sk)*", p = "*string(p)
				
				R1, R2 = rgks(A, numericalRank, p, sketch = sk)
				err = opnorm(A - R1*R2)*1e8
				
				if(err > 1000)
					@warn("rqrcp tests: relative error from standard call exceeds 1000")
					@info params
				end
				
				showInfo(params, @test size(R1) == (size(A, 1), numericalRank))
				showInfo(params, @test size(R2) == (numericalRank, size(A, 2)))
				
				perm = rgks(A, numericalRank, p, format = "minimal", sketch = sk)
				Q = Matrix(qr(A[:, perm]).Q)
				err = opnorm(A - Q*Q'*A)*1e8
				
				if(err > 1000)
					@warn("rqrcp tests: relative error from minimal call exceeds 1000")
					@info params
				end
				
				showInfo(params, @test length(perm) == numericalRank)
				
				R1, R2, perm = rgks(A, numericalRank, p, format = "full", sketch = sk)
				err = opnorm(A - R1*R2)*1e8
				
				if(err > 1000)
					@warn("rqrcp tests: relative error from full call exceeds 1000")
					@info params
				end
				
				showInfo(params, @test size(R1) == (size(A, 1), numericalRank))
				showInfo(params, @test size(R2) == (numericalRank, size(A, 2)))
				showInfo(params, @test length(perm) == numericalRank)
				
				# testing orthonormalization
				R1, R2 = rgks(A, numericalRank, p, sketch = sk, orthonormal = true)
				showInfo(params, @test size(R1) == (size(A, 1), numericalRank))
				showInfo(params, @test size(R2) == (numericalRank, size(A, 2)))
				showInfo(params, @test opnorm(R1'*R1 - I(numericalRank)) < 1e-10)
			end
		end
	end
end
