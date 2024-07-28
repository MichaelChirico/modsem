#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include "mvnorm.h"


// [[Rcpp::depends("RcppArmadillo", "RcppParallel")]]


using namespace RcppParallel;


static double const log2pi = std::log(2.0 * M_PI);


void inplace_tri_mat_mult(arma::rowvec &x, arma::mat const &trimat){
  arma::uword const n = trimat.n_cols;
  
  for(unsigned j = n; j-- > 0;){
    double tmp(0.);
    for(unsigned i = 0; i <= j; ++i)
      tmp += trimat.at(i, j) * x[i];
    x[j] = tmp;
  }
}


arma::vec dmvnrm_arma_mc(const arma::mat &x, 
                         const arma::rowvec &mean, 
                         const arma::mat &sigma, 
                         bool logd = true) {
  using arma::uword;
  uword const n = x.n_rows, 
              xdim = x.n_cols;
  arma::vec out(n);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())), 
               constants = -(double)xdim / 2.0 * log2pi, 
               other_terms = rootisum + constants;
  
  arma::rowvec z;
  for (uword i = 0; i < n; i++) {
    z = (x.row(i) - mean);
    inplace_tri_mat_mult(z, rooti);   
    out(i) = other_terms - 0.5 * arma::dot(z, z);     
  }  
  
  if (logd) {
    return out;
  }
  return exp(out);
}

struct RepDmvnormWorker : public Worker {
  const arma::mat &x;
  const arma::mat &expected;
  const arma::mat &sigma;
  int ncolSigma;
  arma::vec &out;
  
  RepDmvnormWorker(const arma::mat &x, const arma::mat &expected, 
                   const arma::mat &sigma, int ncolSigma, arma::vec &out)
    : x(x), expected(expected), sigma(sigma), ncolSigma(ncolSigma), out(out) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      int firstRow = i * ncolSigma;
      int lastRow = (i + 1) * ncolSigma - 1;
      int firstCol = 0;
      int lastCol = ncolSigma - 1;

      out(i) = dmvnrm_arma_mc(x.row(i), expected.row(i), 
                              sigma.submat(firstRow, firstCol, lastRow, lastCol), true)(0);
    }
  }
};


// [[Rcpp::export]]
arma::vec rep_dmvnorm(const arma::mat &x, 
                      const arma::mat &expected, 
                      const arma::mat &sigma, 
                      const int t) {
  arma::vec out(t);
  int ncolSigma = sigma.n_cols;

  RepDmvnormWorker worker(x, expected, sigma, ncolSigma, out);
  parallelFor(0, t, worker);

  return out;
}
