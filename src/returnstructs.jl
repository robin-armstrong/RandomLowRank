# struct, iterator, and show method for randomized SVD

struct RSVD
	U::Matrix
	S::Vector
	V::Matrix
end

Base.iterate(F::RSVD) = (F.U, Val(:S))
Base.iterate(F::RSVD, ::Val{:S}) = (F.S, Val(:V))
Base.iterate(F::RSVD, ::Val{:V}) = (F.V, Val(:done))
Base.iterate(F::RSVD, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::RSVD)
	summary(io, F); println(io)
	println(io, "U factor:")
	show(io, mime, F.U)
	println(io, "\nsingular values:")
	show(io, mime, F.S)
	println(io, "\nV factor:")
	show(io, mime, F.V)
end

# struct, iterator, and show method for randomized Hermitian eigenvalue decomposition

struct RHEigen
	values::Vector
	vectors::Matrix
end

Base.iterate(F::RHEigen) = (F.values, Val(:vectors))
Base.iterate(F::RHEigen, ::Val{:vectors}) = (F.vectors, Val(:done))
Base.iterate(F::RHEigen, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::RHEigen)
	summary(io, F); println(io)
	println(io, "values:")
	show(io, mime, F.values)
	println(io, "\nvectors:")
	show(io, mime, F.vectors)
end

# struct, iterator, and show method for randomized skeletal decompositions

struct SkeletalDecomp
	perm::Vector
	C::Matrix
	B::Matrix
end

Base.iterate(F::SkeletalDecomp) = (F.perm, Val(:C))
Base.iterate(F::SkeletalDecomp, ::Val{:C}) = (F.C, Val(:B))
Base.iterate(F::SkeletalDecomp, ::Val{:B}) = (F.B, Val(:done))
Base.iterate(F::SkeletalDecomp, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::SkeletalDecomp)
	summary(io, F); println(io)
	println(io, "skeleton column indices:")
	show(io, mime, F.perm)
	println(io, "\nskeleton columns:")
	show(io, mime, F.C)
	println(io, "\ninterpolation matrix:")
	show(io, mime, F.B)
end

# struct, iterator, and show method for orthonormalized skeletal decompositions

struct OrthoSkeletalDecomp
	perm::Vector
	C::Matrix
	B::Matrix
end

Base.iterate(F::OrthoSkeletalDecomp) = (F.perm, Val(:C))
Base.iterate(F::OrthoSkeletalDecomp, ::Val{:C}) = (F.C, Val(:B))
Base.iterate(F::OrthoSkeletalDecomp, ::Val{:B}) = (F.B, Val(:done))
Base.iterate(F::OrthoSkeletalDecomp, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::OrthoSkeletalDecomp)
	summary(io, F); println(io)
	println(io, "skeleton column indices:")
	show(io, mime, F.perm)
	println(io, "\northonormalized skeleton columns:")
	show(io, mime, F.C)
	println(io, "\ninterpolation matrix:")
	show(io, mime, F.B)
end
