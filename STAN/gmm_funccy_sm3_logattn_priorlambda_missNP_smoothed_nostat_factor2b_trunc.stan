/* ***********************************************
 * Use original formulation of CY14 for magnitude scaling
 *********************************************** */

#include functions.stan

data {
  int<lower=1> N;
  int<lower=1> NEQ;
  int<lower=1> NEQ_lm;
  int<lower=1> NEQ_sm;
  int<lower=1> NSTAT;
  int<lower=1> NP;
  int<lower=1> D;
  int D_lt;
  int<lower=1,upper=NP> Pm; // period index for which all records are observed

  real Y_trunc;

  vector[NEQ] M;
  vector[N] Rrup;
  //vector[N] Rjb;
  //vector[N] Rhyp;
  //vector[N] Rx;
  //vector[NEQ] Ztor;
  //vector[NEQ] Zhyp;
  //vector[NEQ] Dip;
  //vector[NEQ] W;
  vector[NSTAT] VS;
  array[N] vector[NP] Y;    // log psa values - 1 is PGA

  array[NP] real lnT; // logarthmic periods

  array[N] int<lower=1,upper=NEQ> eq;
  array[NEQ_lm] int<lower=1,upper=NEQ> eq_lm;
  array[NEQ_sm] int<lower=1,upper=NEQ> eq_sm;
  array[NEQ] int<lower=0,upper=1> idx_sm;
  array[N] int<lower=1,upper=NSTAT> stat;

  array[NP] int<lower=0,upper=N> len_per; // umber of useable records
  array[N, NP] int<lower=0,upper=N> idx_per; // indices of useable records

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

  vector[D_lt] mu_lambda_eq_lm;
  vector<lower=0>[D_lt] sigma_lambda_eq_lm;
  vector[D] mu_ln_lambda_eq_lm_diag;
  vector<lower=0>[D] sigma_ln_lambda_eq_lm_diag;
  vector[D_lt] mu_lambda_rec;
  vector<lower=0>[D_lt] sigma_lambda_rec;
  vector[D] mu_ln_lambda_rec_diag;
  vector<lower=0>[D] sigma_ln_lambda_rec_diag;

  real mu_ln_phi_0;
  real<lower=0> sigma_ln_phi_0;

  real<lower=0> rho;
}

transformed data {
  real delta = 1e-9;
  //int D_lt = D * (NP - D) + (D * (D - 1)) %/% 2;

  // parameters for geomatrical spreading
  real gs_b = log(50.);
  real c4a = -0.5;
  real crb_sq = square(50);

  vector[NP] log_phi4 = log(phi4);

  vector[NSTAT] VS2 = fmin(VS, 1130.);

  // linear predictors
  matrix[N, NP] M_chm;
  vector[N] lnVS_lin = fmin(log(VS[stat] / 1130), 0.);
  matrix[N, NP] VS_nl;
  vector[N] lnR = log(sqrt(square(Rrup) + crb_sq));
  vector[N] lnR_c4a = c4a * lnR;
  matrix[N, NP] attn_M; // magnitude dependent attenuation

  array[NP] matrix[N, 5] X;

  for(p in 1:NP) {
    M_chm[:, p] = fmax(M[eq] - chm[p], 0.);
    VS_nl[:, p] = phi2[p] * (exp(phi3[p] * (VS2[stat] - 360)) - exp(phi3[p] * 770));

    for(i in 1:N) {
      X[p, i, 1] = 1;

      X[p, i, 2] = M[eq[i]] - 6 + log1p_exp(c_n[p] * (c_m[p] - M[eq[i]])) / c_n[p];
      X[p, i, 3] = -1.0 * log1p_exp(c_n[p] * (c_m[p] - M[eq[i]])) / c_n[p];

      X[p, i, 4] = -1.0 * lnR[i];
      X[p, i, 5] = -1.0 * Rrup[i];

      attn_M[i, p] = gamma2[p] / (cosh(fmax(M[eq[i]] - gamma3[p], 0.))) * Rrup[i];
    }
  }

  matrix[NP, NP] L_c1 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_c1[1], rho), delta));
  matrix[NP, NP] L_c2 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_c2[1], rho), delta));
  matrix[NP, NP] L_c5 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_c5[1], rho), delta));
  matrix[NP, NP] L_c6 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_c6[1], rho), delta));
  matrix[NP, NP] L_attn_1 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_c_attn_1_ln[1], rho), delta));
  matrix[NP, NP] L_phi1 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_phi1[1], rho), delta));
  matrix[NP, NP] L_phi_0 = cholesky_decompose(add_diag(gp_exp_quad_cov(lnT, sigma_ln_phi_0, rho), delta));

}

parameters {
  vector[NP] z_c1;
  vector[NP] z_c2;
  vector[NP] z_c5;
  vector[NP] z_c6;
  vector[NP] z_c_attn_1;
  vector[NP] z_phi1;
  real<upper=0> c4;
  
  vector[NP] z_phi_0;

  vector[D_lt] lambda_eq_lm; // Lower diagonal loadings
  vector<lower=0>[D] lambda_eq_lm_diag; // Lower diagonal loadings
  vector[D_lt] lambda_eq_sm; // Lower diagonal loadings
  vector<lower=0>[D] lambda_eq_sm_diag; // Lower diagonal loadings
  vector[D_lt] lambda_rec; // Lower diagonal loadings
  vector<lower=0>[D] lambda_rec_diag; // Lower diagonal loadings

  matrix[D, NEQ_lm] eqterm_fac_lm;
  matrix[D, NEQ_sm] eqterm_fac_sm;
}

transformed parameters {
  matrix[NP, D] A_eq_lm = to_fac_load_lower_tri(lambda_eq_lm, lambda_eq_lm_diag, NP, D); // factor loading matrix
  matrix[NP, D] A_eq_sm = to_fac_load_lower_tri(lambda_eq_sm, lambda_eq_sm_diag, NP, D); // factor loading matrix
  matrix[NP, D] A_rec = to_fac_load(lambda_rec, lambda_rec_diag, NP, D); // factor loading matrix

  matrix[NEQ, NP] eqterm;
  eqterm[eq_lm] = (A_eq_lm * eqterm_fac_lm)';
  eqterm[eq_sm] = (A_eq_sm * eqterm_fac_sm)';

  vector[NP] c1 = mu_c1 + L_c1 * z_c1;
  vector[NP] c2 = mu_c2 + L_c2 * z_c2;
  vector[NP] c5 = mu_c5 + L_c5 * z_c5;
  vector[NP] c6 = mu_c6 + L_c6 * z_c6;
  vector[NP] c_attn_1 = exp(mu_c_attn_1_ln + L_attn_1 * z_c_attn_1);
  vector[NP] phi1 = mu_phi1 + L_phi1 * z_phi1;
  vector[NP] phi_0 = exp(mu_ln_phi_0 + L_phi_0 * z_phi_0);
}

model {
  array[N] vector[NP] mu_rec;

  // prior distributions for coefficients
  z_c1 ~ std_normal();
  z_c2 ~ std_normal();
  z_c5 ~ std_normal();
  z_c6 ~ std_normal();
  z_phi1 ~ std_normal();
  z_c_attn_1 ~ std_normal();
  z_phi_0 ~ std_normal();
  c4 ~ normal(mu_c4, sigma_c4);

  lambda_eq_lm ~ normal(mu_lambda_eq_lm, sigma_lambda_eq_lm);
  lambda_eq_lm_diag ~ lognormal(mu_ln_lambda_eq_lm_diag, sigma_ln_lambda_eq_lm_diag);
  lambda_eq_sm ~ normal(mu_lambda_eq_lm, sigma_lambda_eq_lm);
  lambda_eq_sm_diag ~ lognormal(mu_ln_lambda_eq_lm_diag, sigma_ln_lambda_eq_lm_diag);
  lambda_rec ~ normal(mu_lambda_rec, sigma_lambda_rec);
  lambda_rec_diag ~ lognormal(mu_ln_lambda_rec_diag, sigma_ln_lambda_rec_diag);

  to_vector(eqterm_fac_lm) ~ std_normal();
  to_vector(eqterm_fac_sm) ~ std_normal();

  for(p in 1:NP) {
    vector[N] mu_lin = X[p] * [c1[p], c2[p], c3[p], c4, c_attn_1[p]]' + lnR_c4a + attn_M[:,p] + eqterm[eq, p];

    // data likelihood
    for(i in 1:N) {
      real fgs = c4 * log(Rrup[i] + c5[p] * cosh(c6[p] * M_chm[i, p]));

      // calculating yref
      real yref = mu_lin[i] + fgs;

      mu_rec[i, p] = yref + phi1[p] * lnVS_lin[i] + VS_nl[i, p] * log1p_exp(yref - log_phi4[p]);
    }
  }
  matrix[NP, NP] L_Sigma = cholesky_decompose(add_diag(A_rec * A_rec', square(phi_0)));
  real phi_pga = sqrt(square(lambda_rec_diag[1]) + square(phi_0[1]));
  // observation likelihood
  target += multi_normal_cholesky_lpdf(Y[idx_per[1:len_per[1],1]] | mu_rec[idx_per[1:len_per[1],1]], L_Sigma) - normal_lccdf(Y_trunc | mu_rec[idx_per[1:len_per[1],1],1], phi_pga);
  for(p in Pm:(NP - 1)) {
    target += multi_normal_cholesky_lpdf(Y[idx_per[1:len_per[p],p], 1:p] | mu_rec[idx_per[1:len_per[p],p], 1:p], block(L_Sigma, 1, 1, p, p)) - normal_lccdf(Y_trunc | mu_rec[idx_per[1:len_per[p],p],1], phi_pga);
  }
}

generated quantities {
  matrix[NP,NP] C_eq_lm = cov2cor(A_eq_lm * A_eq_lm');
  matrix[NP,NP] C_eq_sm = cov2cor(A_eq_sm * A_eq_sm');
  matrix[NP,NP] C_rec = cov2cor(A_rec * A_rec');
  vector[NP] tau = sqrt(diagonal(A_eq_lm * A_eq_lm'));
  vector[NP] tau_sm = sqrt(diagonal(A_eq_sm * A_eq_sm'));
  vector[NP] phi = sqrt(diagonal(A_rec * A_rec') + square(phi_0));
}
