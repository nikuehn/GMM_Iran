/* ***********************************************
 * Use original formulation of CY14 for magnitude scaling
 *********************************************** */

#include functions.stan

data {
  int<lower=1> N;
  int<lower=1> NP;

  vector[N] M;
  vector[N] Rrup;
  //vector[N] Rjb;
  //vector[N] Rx;
  //vector[NEQ] Ztor;
  //vector[NEQ] Zhyp;
  //vector[NEQ] Dip;
  //vector[NEQ] W;
  vector[N] VS;

  // fixed coefficients
  vector[NP] c_m;
  vector[NP] c_n;
  vector[NP] c3;
  vector[NP] chm;
  vector[NP] phi2;
  vector[NP] phi3;
  vector[NP] phi4;
  vector[NP] gamma2;
  vector[NP] gamma3;

  //vector[NP] chw1;
  //vector[NP] chw2;
  //vector[NP] chw3;
  //vector[NP] chw4;

}

transformed data {
  real delta = 1e-9;

  // parameters for geomatrical spreading
  real gs_b = log(50.);
  real c4a = -0.5;
  real crb_sq = square(50);

  vector[NP] log_phi4 = log(phi4);

  vector[N] VS2 = fmin(VS, 1130.);

  // linear predictors
  matrix[N, NP] M_chm;
  vector[N] lnVS_lin = fmin(log(VS / 1130), 0.);
  matrix[N, NP] VS_nl;
  matrix[N, NP] attn_M; // magnitude dependent attenuation

  array[NP] matrix[N, 3] X;

  for(p in 1:NP) {
    M_chm[:, p] = fmax(M - chm[p], 0.);
    VS_nl[:, p] = phi2[p] * (exp(phi3[p] * (VS2 - 360)) - exp(phi3[p] * 770));

    for(i in 1:N) {
      X[p, i, 1] = 1;

      X[p, i, 2] = M[i] - 6 + log1p_exp(c_n[p] * (c_m[p] - M[i])) / c_n[p];
      X[p, i, 3] = -1.0 * log1p_exp(c_n[p] * (c_m[p] - M[i])) / c_n[p];

      attn_M[i, p] = gamma2[p] / (cosh(fmax(M[i] - gamma3[p], 0.)));
    }
  }

}

parameters {
  vector[NP] c1;
  vector[NP] c2;
  vector[NP] c5;
  vector[NP] c6;
  vector[NP] c_attn_1;
  vector[NP] phi1;
  real c4;
}

model {
}

generated quantities {
  array[N] vector[NP] mu_pred;
  for(p in 1:NP) {
    vector[N] mu_lin = X[p] * [c1[p], c2[p], c3[p]]';

    // data likelihood
    for(i in 1:N) {
      real lnR = log(Rrup[i] + c5[p] * cosh(c6[p] * fmax(M[i] - chm[p], 0.)));
      real fgs = c4 * lnR + (c4a - c4) * log(sqrt(square(Rrup[i]) + crb_sq));

      // calculating yref
      real yref = mu_lin[i] + fgs - c_attn_1[p] * Rrup[i] + attn_M[i,p] * Rrup[i];

      mu_pred[i, p] = yref + phi1[p] * lnVS_lin[i] + VS_nl[i, p] * log1p_exp(yref - log_phi4[p]);
    }
  }
}
