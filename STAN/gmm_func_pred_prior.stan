/* ***********************************************
 *********************************************** */

#include functions.stan

data {
  int<lower=1> N;
  int<lower=1> NP;

  vector[N] M;
  vector[N] Rrup;
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

  // global parameters
  vector[NP] mu_c1;
  vector[NP] mu_c2;
  vector[NP] mu_c5;
  vector[NP] mu_c6;
  vector[NP] mu_c_attn_1_ln;
  vector[NP] mu_phi1;
  real mu_c4;

  vector[NP] sigma_c1;
  vector[NP] sigma_c2;
  vector[NP] sigma_c5;
  vector[NP] sigma_c6;
  vector[NP] sigma_c_attn_1_ln;
  vector[NP] sigma_phi1;
  real sigma_c4;

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

  array[NP] matrix[N, 4] X;

  for(p in 1:NP) {
    M_chm[:, p] = fmax(M - chm[p], 0.);
    VS_nl[:, p] = phi2[p] * (exp(phi3[p] * (VS2 - 360)) - exp(phi3[p] * 770));

    for(i in 1:N) {
      X[p, i, 1] = 1;

      X[p, i, 2] = M[i] - 6 + log1p_exp(c_n[p] * (c_m[p] - M[i])) / c_n[p];
      X[p, i, 3] = -1.0 * log1p_exp(c_n[p] * (c_m[p] - M[i])) / c_n[p];

      X[p, i, 4] = -1.0 * Rrup[i];

      attn_M[i, p] = gamma2[p] / (cosh(fmax(M[i] - gamma3[p], 0.))) * Rrup[i];
    }
  }
}

parameters {
}

model {
}

generated quantities {
  array[N] vector[NP] mu_rec;

  for(p in 1:NP) {

    real c1 = normal_rng(mu_c1[p], sigma_c1[p]);
    real c2 = normal_rng(mu_c2[p], sigma_c2[p]);
    real c5 = normal_rng(mu_c5[p], sigma_c5[p]);
    real c6 = normal_rng(mu_c6[p], sigma_c6[p]);
    real c_attn_1 = lognormal_rng(mu_c_attn_1_ln[p], sigma_c_attn_1_ln[p]);
    real phi1 = normal_rng(mu_phi1[p], sigma_phi1[p]);
    real c4 = normal_rng(mu_c4, sigma_c4);

    vector[N] mu_lin = X[p] * [c1, c2, c3[p], c_attn_1]' + attn_M[:,p];
    // data likelihood
    for(i in 1:N) {
      real lnR = log(Rrup[i] + c5 * cosh(c6 * fmax(M[i] - chm[p], 0.)));
      real fgs = c4 * lnR + (c4a - c4) * log(sqrt(square(Rrup[i]) + crb_sq));

      // calculating yref
      real yref = mu_lin[i] + fgs;

      mu_rec[i, p] = yref + phi1 * lnVS_lin[i] + VS_nl[i, p] * log1p_exp(yref - log_phi4[p]);
    }
  }

}
