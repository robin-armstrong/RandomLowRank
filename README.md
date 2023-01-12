# RandomLowRank

This package provides Julia implementations of randomized algorithms for constructing approximate low-rank factorizations of large matrices. The package can be installed by running the following commands in Julia:
```
import Pkg
Pkg.add(url="https://github.com/robin-armstrong/RandomLowRank.git")
```

After installation, run `using RandomLowRank` to bring the package's exported names into the namespace.

### Functions

This package offers the following functions to compute standard matrix factorizations:

* `rsvd`, a randomized singular value decomposition, and
* `rheigen`, a randomized symmetric eigenvalue decomposition,

both of which are computed using the techniques of Halko, Martinsson, and Tropp (2011). It also offers functions to compute interpolative decompositions, i.e. approximate factorizations of the form `A = C*B` where `C` consists of selected "skeleton" columns from `A` and `B` is an interpolation matrix. Specifically, the package offers

* `rgks`, which selects skeleton columns using a randomized version of the algorithm of Golub, Klemma, and Stewart (1976), and
* `rqrcp`, which selects skeleton columns using a randomized column-pivoted QR factorization.

More information on these functions can be found by running `?function-name` in a Julia REPL.

### Sketches

Each of the functions in `RandomLowRank` begins by computing a "sketch" of the input matrix `A`, i.e. by computing `S = A*Omega` or `S = Omega'*A`, where `Omega` is a random matrix whose dimensions are much smaller than that of `A`. Extensive information on this process can be found in Halko, Martinsson, and Tropp (2011). The `RandomLowRank` package allows the user to specify the process by which `Omega` is generated; this is accomplished by passing a `Sketch` object to the function being used. `RandomLowRank` offers four types of `Sketch` objects:

* `GaussianSketch <: Sketch`, which specifies that `Omega` should have i.i.d. standard Gaussian entries. This is the default which is used when the user does not specify a sketch.
* `HadamardSketch <: Sketch`, which specifies that `Omega` is a subsampled randomized Hadamard transform (see Tropp, "Improved Analysis of the Subsampled Randomized Hadamard Transform", 2010).
* `CWSketch <: Sketch`, which specifies that `Omega` is a sparse Clarkson-Woodruff sketching matrix (see Clarkson and Woodruff, "Low-Rank Approximation and Regression in Input Sparsity Time", 2013).
* `NoSketch <: Sketch`, which specifies that the full input matrix should be used without any sketching. This should be used when the user wishes to run the algorithms of `RandomLowRank` deterministically.

`Sketch` objects are passed using the keyword argument `sk`. For example, to compute a rank `k` approximation to a matrix `A` using a randomized SVD with Clarkson-Woodruff sketching, run
```
U, S, Vt = rsvd(A, k, sk = CWSketch())
```
 
Custom sketches can be implemented by defining a concrete sub-type of `Sketch` and then implementing the corresponding method for the function `sketch` function. For more information on this process, run `?Sketch` in a Julia REPL.
