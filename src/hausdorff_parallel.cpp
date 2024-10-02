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
