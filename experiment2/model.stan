data {
  int<lower=1> nt; // number of trials in total
  int<lower=1> ns; // number of subjects
  int<lower=1,upper=ns> sid[nt]; // subject index for each trial
  real psp[nt]; // payoff, self, predicted: average PAYOFF for the computer (SELF) for the row PREDICTED by the subject
  real pop[nt]; // payoff, opponent, predicted
  real psn[nt]; // payoff, self, non-predicted
  real pon[nt]; // payoff, opponent, non-predicted
}

parameters {
  real<lower=0> alpha_gamma_alpha; // alpha parameter of the gamma distribution from which the softmax parameters are drawn
  real<lower=0> alpha_gamma_beta; // beta parameter of ...
  real<lower=0> alpha[ns]; // softmax parameter for each subject

  real lambda_normal_mu; // mean of the normal distribution from which lambdas are drawn
  real<lower=0> lambda_normal_sigma; // sd of ...
  real lambda[ns]; // lambda for each subject
}

model {
  real log_odds[nt]; // log-odds that the computer chooses the predicted row (Bernoulli parameter)

  for (i in 1:nt) {
    real l = lambda[sid[i]]; // lambda
    real up = psp[i] + l * pop[i]; // utility for the predicted row
    real un = psn[i] + l * pon[i]; // utility for the non-predicted row
    
    log_odds[i] = (up - un) * alpha[sid[i]];
  }

  alpha_gamma_alpha ~ exponential(1);
  alpha_gamma_beta ~ exponential(1);
  alpha ~ gamma(alpha_gamma_alpha, alpha_gamma_beta);

  lambda_normal_sigma ~ exponential(1);
  lambda ~ normal(lambda_normal_mu, lambda_normal_sigma);
  
  1 ~ bernoulli_logit(log_odds);
}
