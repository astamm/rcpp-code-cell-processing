# Including multiple files in a single Rcpp chunk


## Issues and workarounds

### First issue

If you have a habit of separating function declaration in a header file
with `.h` extension and function implementation in the eponymous `.cpp`
file, then including the `cpp` file or the `.h` file in a Rcpp code
chunk will result in failure to compile. This is because Rcpp code
chunks make use of `Rcpp::sourceCpp()` function, which only accepts a
single `.cpp` file by design.

**Solution.** You should convert every pair of file (`myfile.h`,
`myfile.cpp`) into a single file (`myfile.hpp`) and then include the
`.hpp` file in the Rcpp code chunk.

### Second issue

If you have multiple files to include in a single Rcpp code chunk, then
you cannot include them directly in the Rcpp code chunk. This is
because, behind the scene, `Rcpp::sourceCpp()` creates a – possibly
temporary – cache directory into which it copies only the single input
file provided in the `file` argument.

**Solution.** You can manually copy all the files you want to include in
the Rcpp code chunk to the cache directory created by
`Rcpp::sourceCpp()`. This can be done by setting the `cacheDir` argument
of the `Rcpp::sourceCpp()` function to the desired cache directory via
the `cacheDir` optional argument. Behind the scene, `Rcpp::sourceCpp()`
subsequently creates a sub-directory in that directory whose name is
platform-dependent and retrieved via
`Rcpp:::.sourceCppPlatformCacheDir()`. We can therefore decide
beforehand of the location of that cache directory, create it and copy
the files we want to include in the Rcpp code chunk to that directory.

### Do not copy header files!

If you have header files to include in the Rcpp code chunk, you should
not copy as per the first issue above. Instead, you should convert them
to `.hpp` files and include them in the Rcpp code chunk.

## Live example

The following code assumes that all your source files are in the `src`
directory into the directory where your `.qmd` file is located. The code
demonstrates how to include multiple files in a single Rcpp code chunk.

The first and last code chunks are the ones handling the copying of the
files to the cache directory and the deletion of the cache directory.
They can be hidden in the final document by setting the code chunk
option `include` to `false`.

First, we create the cache directory and copy the header file
`hausdorff_utils.hpp` to it.

``` r
cache_dir_base <- "."
cache_dir <- Rcpp:::.sourceCppPlatformCacheDir(cache_dir_base)
fs::dir_create(cache_dir)
fs::file_copy(
  path = list.files("src", pattern = "*.hpp", full.names = TRUE), 
  new_path = cache_dir, 
  overwrite = TRUE
)
```

Then we can compile a `.cpp` file which depends on a `.hpp` file that
has been copied to the cache directory by setting the `cacheDir`
argument of the `Rcpp::sourceCpp()` function to the cache directory that
we curated before.

    # ```{Rcpp, engine.opts = list(cacheDir = ".")}
    # #| file: src/hausdorff_parallel.cpp
    # ```

``` cpp
#include "hausdorff_utils.hpp"

struct HausdorffDistanceComputer : public RcppParallel::Worker
{
  const RcppParallel::RMatrix<double> m_SafeInput;
  RcppParallel::RVector<double> m_SafeOutput;
  unsigned int m_Dimension;

  HausdorffDistanceComputer(const Rcpp::NumericMatrix x,
                            Rcpp::NumericVector out,
                            unsigned int dimension)
    : m_SafeInput(x), m_SafeOutput(out), m_Dimension(dimension) {}

  void operator()(std::size_t begin, std::size_t end)
  {
    unsigned int N = m_SafeInput.nrow();
    for (std::size_t k = begin;k < end;++k)
    {
      unsigned int i = N - 2 - std::floor(std::sqrt(-8 * k + 4 * N * (N - 1) - 7) / 2.0 - 0.5);
      unsigned int j = k + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2;
      m_SafeOutput[k] = hausdorff_distance_cpp(m_SafeInput.row(i), m_SafeInput.row(j), m_Dimension);
    }
  }
};

Rcpp::NumericVector dist_parallel(Rcpp::NumericMatrix x,
                                  unsigned int dimension = 1,
                                  unsigned int ncores = 1)
{
  unsigned int N = x.nrow();
  unsigned int K = N * (N - 1) / 2;
  Rcpp::NumericVector out(K);

  HausdorffDistanceComputer hausdorffDistance(x, out, dimension);
  RcppParallel::parallelFor(0, K, hausdorffDistance, 1, ncores);

  out.attr("Size") = N;
  out.attr("Labels") = Rcpp::seq(1, N);
  out.attr("Diag") = false;
  out.attr("Upper") = false;
  out.attr("method") = "hausdorff";
  out.attr("class") = "dist";
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector dist_parallel(Rcpp::List x,
                                  unsigned int dimension = 1,
                                  unsigned int ncores = 1)
{
  Rcpp::NumericMatrix xMatrix = listToMatrix(x);
  return dist_parallel(xMatrix, dimension, ncores);
}
```

The code above implements the function `dist_parallel()` which performs
pairwise Hausdorff distance computations within a functional data set,
possibly in parallel. The function expects as input a list of size $N$
containing the $N$ $L$-dimensional curves observed on a given grid of
size $M$ of their common domain. Each entry of the input list must be a
matrix of shape $L \times M$. The simulated data stored in
`data/dat.rds` is formatted in such a way. We can therefore check that
we can use the exported function `dist_parallel()` as follows.

``` r
dat <- readRDS("data/dat.rds")
dat <- dat[1:21]
D <- dist_parallel(dat, dimension = 3L)
head(D)
```

    [1] 1.3226988 1.5403096 1.0560611 0.9340005 1.7887889 1.4880584

Finally, we can clean up the cache directory.

``` r
fs::dir_delete(cache_dir)
```
