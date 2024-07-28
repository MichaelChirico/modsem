#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include "QML.h"


// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]


using namespace RcppParallel;


//' @importFrom RcppParallel RcppParallelLibs
struct MuQmlWorker : public Worker {
  const int numEta;
  const int numXi;
  const arma::mat& alpha;
  const arma::mat& beta0;
  const arma::mat& gammaXi;
  const arma::mat& omegaXiXi;
  const arma::mat& l1;
  const arma::mat& l2;
  const arma::mat& x;
  const arma::mat& u;
  const arma::mat& Binv;
  const arma::vec& trOmegaSigma;
  const arma::mat& kronXi;

  arma::mat& Ey;

  MuQmlWorker(int numEta, int numXi, const arma::mat& alpha, const arma::mat& beta0, 
              const arma::mat& gammaXi, const arma::mat& omegaXiXi, const arma::mat& l1, 
              const arma::mat& l2, const arma::mat& x, const arma::mat& u, const arma::mat& Binv,
              const arma::vec& trOmegaSigma, const arma::mat& kronXi, arma::mat& Ey)
    : numEta(numEta), numXi(numXi), alpha(alpha), beta0(beta0), gammaXi(gammaXi), omegaXiXi(omegaXiXi), 
      l1(l1), l2(l2), x(x), u(u), Binv(Binv), trOmegaSigma(trOmegaSigma), kronXi(kronXi), Ey(Ey) {}

  void operator()(std::size_t begin, std::size_t end) override {
    int firstCol = 0;
    int lastColKOxx = numXi * numEta - 1;
    int lastColBinv = numEta - 1;

    for (std::size_t i = begin; i < end; i++) {
      int firstRow = i * numEta;
      int lastRow = (i + 1) * numEta - 1;

      arma::mat kronXi_t = kronXi.submat(firstRow, firstCol, lastRow, lastColKOxx);
      arma::mat Binv_t = int(Binv.n_rows) > numEta ? Binv.submat(firstRow, firstCol, lastRow, lastColBinv) : Binv;

      Ey.row(i) = (Binv_t * (trOmegaSigma + alpha + gammaXi * (beta0 + l1 * x.row(i).t()) +
                             kronXi_t * omegaXiXi * (beta0 + l1 * x.row(i).t())) +
                   l2 * u.row(i).t())
                    .t();
    }
  }
};


// [[Rcpp::export]]
arma::mat muQmlCpp(Rcpp::List m, int t) {
  int numEta = Rcpp::as<int>(m["numEta"]);
  int numXi = Rcpp::as<int>(m["numXi"]);
  arma::mat alpha = Rcpp::as<arma::mat>(m["alpha"]);
  arma::mat beta0 = Rcpp::as<arma::mat>(m["beta0"]);
  arma::mat gammaXi = Rcpp::as<arma::mat>(m["gammaXi"]);
  arma::mat omegaXiXi = Rcpp::as<arma::mat>(m["omegaXiXi"]);
  arma::mat l1 = Rcpp::as<arma::mat>(m["L1"]);
  arma::mat l2 = Rcpp::as<arma::mat>(m["L2"]);
  arma::mat x = Rcpp::as<arma::mat>(m["x"]);
  arma::mat u = Rcpp::as<arma::mat>(m["u"]);
  arma::mat Ey = arma::mat(t, numEta);
  arma::mat Binv = Rcpp::as<arma::mat>(m["Binv"]);
  arma::vec trOmegaSigma = traceOmegaSigma1(omegaXiXi * Rcpp::as<arma::mat>(m["Sigma1"]), numEta);
  arma::mat kronXi = Rcpp::as<arma::mat>(m["kronXi"]);

  MuQmlWorker worker(numEta, numXi, alpha, beta0, gammaXi, omegaXiXi, l1, l2, x, u, Binv, trOmegaSigma, kronXi, Ey);
  parallelFor(0, t, worker);

  return Ey;
}


struct SigmaQmlWorker : public Worker {
  const int numEta;
  const int numXi;
  const arma::mat& gammaXi;
  const arma::mat& omegaXiXi;
  const arma::mat& l1;
  const arma::mat& l2;
  const arma::mat& x;
  const arma::mat& u;
  const arma::mat& Binv;
  const arma::mat& psi;
  const arma::mat& Sigma1;
  const arma::mat& Sigma2ThetaEpsilon;
  const arma::mat& varZ;
  const arma::mat& kronXi;

  arma::mat& sigmaE;

  SigmaQmlWorker(int numEta, int numXi, const arma::mat& gammaXi, const arma::mat& omegaXiXi, 
                 const arma::mat& l1, const arma::mat& l2, const arma::mat& x, const arma::mat& u, 
                 const arma::mat& Binv, const arma::mat& psi, const arma::mat& Sigma1, 
                 const arma::mat& Sigma2ThetaEpsilon, const arma::mat& varZ, const arma::mat& kronXi, 
                 arma::mat& sigmaE)
    : numEta(numEta), numXi(numXi), gammaXi(gammaXi), omegaXiXi(omegaXiXi), l1(l1), l2(l2), x(x), u(u),
      Binv(Binv), psi(psi), Sigma1(Sigma1), Sigma2ThetaEpsilon(Sigma2ThetaEpsilon), varZ(varZ), kronXi(kronXi),
      sigmaE(sigmaE) {}

  void operator()(std::size_t begin, std::size_t end) override {
    int firstCol = 0;
    int lastColSigmaE = numEta - 1;
    int lastColKOxx = numXi * numEta - 1;

    for (std::size_t i = begin; i < end; i++) {
      int firstRow = i * numEta;
      int lastRow = (i + 1) * numEta - 1;

      arma::mat kronXi_t = kronXi.submat(firstRow, firstCol, lastRow, lastColKOxx);

      if (int(Binv.n_rows) > numEta) {
        arma::mat Binv_t = Binv.submat(firstRow, firstCol, lastRow, lastColSigmaE);
        arma::mat Sigma2 = Binv_t * psi * Binv_t.t() + Sigma2ThetaEpsilon;
        sigmaE.submat(firstRow, firstCol, lastRow, lastColSigmaE) = 
          (Binv_t * (gammaXi + 2 * kronXi_t * omegaXiXi)) * Sigma1 * 
          (Binv_t * (gammaXi + 2 * kronXi_t * omegaXiXi)).t() + Sigma2 + 
          Binv_t * varZ * Binv_t.t();
      } else {
        arma::mat varZ_t = Binv * varZ * Binv.t();
        arma::mat Sigma2 = Binv * psi * Binv.t() + Sigma2ThetaEpsilon;
        sigmaE.submat(firstRow, firstCol, lastRow, lastColSigmaE) = 
          (Binv * (gammaXi + 2 * kronXi_t * omegaXiXi)) * Sigma1 * 
          (Binv * (gammaXi + 2 * kronXi_t * omegaXiXi)).t() + Sigma2 + varZ_t;
      }
    }
  }
};


// [[Rcpp::export]]
arma::mat sigmaQmlCpp(Rcpp::List m, int t) {
  int numEta = Rcpp::as<int>(m["numEta"]);
  int numXi = Rcpp::as<int>(m["numXi"]);
  arma::mat gammaXi = Rcpp::as<arma::mat>(m["gammaXi"]);
  arma::mat omegaXiXi = Rcpp::as<arma::mat>(m["omegaXiXi"]);
  arma::mat l1 = Rcpp::as<arma::mat>(m["L1"]);
  arma::mat l2 = Rcpp::as<arma::mat>(m["L2"]);
  arma::mat x = Rcpp::as<arma::mat>(m["x"]);
  arma::mat u = Rcpp::as<arma::mat>(m["u"]);
  arma::mat Sigma1 = Rcpp::as<arma::mat>(m["Sigma1"]);
  arma::mat Sigma2ThetaEpsilon = Rcpp::as<arma::mat>(m["Sigma2ThetaEpsilon"]);
  arma::mat psi = Rcpp::as<arma::mat>(m["psi"]);
  arma::mat Ie = Rcpp::as<arma::mat>(m["Ieta"]);
  arma::mat sigmaE = arma::mat(t * numEta, numEta);
  arma::mat Binv = Rcpp::as<arma::mat>(m["Binv"]);
  arma::mat varZ = varZCpp(omegaXiXi, Sigma1, numEta); 
  arma::mat kronXi = Rcpp::as<arma::mat>(m["kronXi"]);

  SigmaQmlWorker worker(numEta, numXi, gammaXi, omegaXiXi, l1, l2, x, u, Binv, psi, Sigma1, Sigma2ThetaEpsilon, varZ, kronXi, sigmaE);
  parallelFor(0, t, worker);

  return sigmaE;
}


struct CalcKronXiWorker : public Worker {
  const int numEta;
  const int numXi;
  const arma::mat& beta0;
  const arma::mat& l1;
  const arma::mat& x;
  const arma::mat& Ie;

  arma::mat& out;

  CalcKronXiWorker(int numEta, int numXi, const arma::mat& beta0, const arma::mat& l1, 
                   const arma::mat& x, const arma::mat& Ie, arma::mat& out)
    : numEta(numEta), numXi(numXi), beta0(beta0), l1(l1), x(x), Ie(Ie), out(out) {}

  void operator()(std::size_t begin, std::size_t end) override {
    for (std::size_t i = begin; i < end; i++) {
      out.submat(i * numEta, 0, (i + 1) * numEta - 1, numXi * numEta - 1) = 
        arma::kron(Ie, beta0.t() + x.row(i) * l1.t());
    }
  }
};


// [[Rcpp::export]]
arma::mat calcKronXi(Rcpp::List m, int t) {
  int numEta = Rcpp::as<int>(m["numEta"]);  
  int numXi = Rcpp::as<int>(m["numXi"]);
  arma::mat beta0 = Rcpp::as<arma::mat>(m["beta0"]);
  arma::mat l1 = Rcpp::as<arma::mat>(m["L1"]); 
  arma::mat x = Rcpp::as<arma::mat>(m["x"]);
  arma::mat Ie = Rcpp::as<arma::mat>(m["Ieta"]);
  
  arma::mat out = arma::mat(t * numEta, numXi * numEta);

  CalcKronXiWorker worker(numEta, numXi, beta0, l1, x, Ie, out);

  parallelFor(0, t, worker);

  return out;
}


struct CalcBinvWorker : public Worker {
  const int numEta;
  const int numXi;
  const arma::mat& B;
  const arma::mat& omegaEtaXi;
  const arma::mat& kronXi;

  arma::mat& B_t;

  CalcBinvWorker(int numEta, int numXi, const arma::mat& B, const arma::mat& omegaEtaXi, 
                 const arma::mat& kronXi, arma::mat& B_t)
    : numEta(numEta), numXi(numXi), B(B), omegaEtaXi(omegaEtaXi), kronXi(kronXi), B_t(B_t) {}

  void operator()(std::size_t begin, std::size_t end) override {
    int firstCol = 0;
    int lastColB = numEta - 1;
    int lastColKOxx = numXi * numEta - 1;

    for (std::size_t i = begin; i < end; i++) {
      int firstRow = i * numEta;
      int lastRow = (i + 1) * numEta - 1;
      
      arma::mat kronXi_t = 
        kronXi.submat(firstRow, firstCol, lastRow, lastColKOxx);

      B_t.submat(firstRow, firstCol, lastRow, lastColB) = 
        arma::inv(B - kronXi_t * omegaEtaXi);
    }
  }
};


// [[Rcpp::export]]
arma::mat calcBinvCpp(Rcpp::List m, int t) {
  int numEta = Rcpp::as<int>(m["numEta"]); 
  int numXi = Rcpp::as<int>(m["numXi"]);
  int kOmegaEta = Rcpp::as<int>(m["kOmegaEta"]);
  arma::mat gammaEta = Rcpp::as<arma::mat>(m["gammaEta"]);

  arma::mat Ie = Rcpp::as<arma::mat>(m["Ieta"]); 
  arma::mat B = Ie - gammaEta;
  arma::mat omegaEtaXi = Rcpp::as<arma::mat>(m["omegaEtaXi"]);

  if (numEta == 1) return Ie;
  else if (kOmegaEta == 0) return arma::inv(B);

  arma::mat kronXi = Rcpp::as<arma::mat>(m["kronXi"]);
  arma::mat B_t = arma::mat(t * numEta, numEta);

  CalcBinvWorker worker(numEta, numXi, B, omegaEtaXi, kronXi, B_t);

  parallelFor(0, t, worker);

  return B_t;
}


struct LogNormalPdfWorker : public Worker {
  const arma::vec& x;
  const arma::vec& mu;
  const arma::mat& sigma;
  arma::vec& result;

  LogNormalPdfWorker(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma, arma::vec& result)
    : x(x), mu(mu), sigma(sigma), result(result) {}

  void operator()(std::size_t begin, std::size_t end) {
    double log_2pi = std::log(2.0 * M_PI);
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < sigma.n_cols; j++) {
        double diff = x(i) - mu(i);
        double sigma_sq = sigma(i, j) * sigma(i, j);
        result(i) += -0.5 * log_2pi - std::log(sigma(i, j)) - 0.5 * (diff * diff) / sigma_sq;
      }
    }
  }
};

arma::vec logNormalPdfParallel(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma) {
  int n = x.n_elem;
  arma::vec result(n, arma::fill::zeros);

  LogNormalPdfWorker worker(x, mu, sigma, result);
  parallelFor(0, n, worker);

  return result;
}

// [[Rcpp::export]]
arma::vec dnormCpp(const arma::vec& x, const arma::vec& mu, const arma::vec& sigma) {
  return logNormalPdfParallel(x, mu, sigma);
}


// [[Rcpp::export]]
arma::mat varZCpp(arma::mat Omega, arma::mat Sigma1, int numEta) {
  arma::mat varZ = arma::mat(numEta, numEta);
  int subRows = Omega.n_rows / numEta; 
  for (int i = 0; i < numEta; i++) {
    varZ(i, i) = varZSubOmega(Omega.submat(i * subRows, 0,
          (i + 1) * subRows - 1, (Omega.n_cols - 1)), Sigma1);
  }
  return varZ;
}


double varZSubOmega(arma::mat Omega, arma::mat Sigma1) {

  int ds = Sigma1.n_rows;
  double varZ = 0;
  
  for (int i = 0; i < ds; i++) {
    for (int j = 0; j < ds; j++) {
      for (int k = 0; k < ds; k++) {
        for (int s = 0; s < ds; s++) {
          varZ += Omega(i, j) * Omega(k, s) * 
            (Sigma1(i, j) * Sigma1(k, s) + 
             Sigma1(i, k) * Sigma1(j, s) +
             Sigma1(i, s) * Sigma1(j, k));
        }
      }
    }
  }
  double trOmegaSigma1 = arma::trace(Omega * Sigma1);
  return varZ - trOmegaSigma1 * trOmegaSigma1;
}


arma::vec traceOmegaSigma1(const arma::mat OmegaSigma1, const int numEta) {
  arma::vec trace = arma::vec(numEta);
  int subRows = OmegaSigma1.n_rows / numEta;
  for (int i = 0; i < numEta; i++) {
    for (int j = 0; j < int(OmegaSigma1.n_cols); j++) {
      trace(i) += OmegaSigma1(i * subRows + j, j);
    } 
  }
  return trace;
}
