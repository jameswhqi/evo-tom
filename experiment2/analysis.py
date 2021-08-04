#%%
import json
import os
import os.path

from funcy import lmap, merge, lcat
from cmdstanpy import CmdStanModel
import numpy as np
import statsmodels.api as sm

#%%
def load_data(path):
    with open(path, 'r') as fh:
        return json.load(fh)

def load_all(directory: str):
    paths = (os.path.join(directory, f) for f in os.listdir(directory) if f.startswith('sona'))
    return lmap(load_data, paths)

def get_payoffs(trial):
    p = trial['payoffs']
    if trial['prediction'] == 0:
        return {
            'psp': (p[0] + p[2]) / 2,
            'pop': (p[1] + p[3]) / 2,
            'psn': (p[4] + p[6]) / 2,
            'pon': (p[5] + p[7]) / 2
        }
    else:
        return {
            'psp': (p[4] + p[6]) / 2,
            'pop': (p[5] + p[7]) / 2,
            'psn': (p[0] + p[2]) / 2,
            'pon': (p[1] + p[3]) / 2
        }

def get_lambdas(summary):
    return summary.loc[np.char.startswith(summary.index.to_numpy(dtype='str'), 'lambda['), 'Mean'].to_numpy()

#%%
subjects = load_all('data')
subjectss = [
    [s for s in subjects if s['client']['lambda'] == l]
    for l in [-1, 0, 1]
]
trialss = [
    lcat([
        [merge({ 'sid': i + 1 }, get_payoffs(t)) for t in s['trials']]
        for i, s in enumerate(ss)
    ])
    for ss in subjectss
]
data = [
    merge(
        { 'nt': len(ts), 'ns': len(ss) },
        { k: [dic[k] for dic in ts] for k in ts[0] }
    )
    for ss, ts in zip(subjectss, trialss)
]

#%%
model = CmdStanModel(stan_file='model.stan')
fitn1 = model.sample(data=data[0], parallel_chains=1)
fit0 = model.sample(data=data[1], parallel_chains=1)
fit1 = model.sample(data=data[2], parallel_chains=1)
summaryn1 = fitn1.summary(sig_figs=4)
summary0 = fit0.summary(sig_figs=4)
summary1 = fit1.summary(sig_figs=4)

#%%
print('Posterior of mu for lambda = -1:')
print('  Mean:', summaryn1.loc['lambda_normal_mu', 'Mean'])
print('  SD:', summaryn1.loc['lambda_normal_sigma', 'StdDev'])
print()
print('Posterior of mu for lambda = 0:')
print('  Mean:', summary0.loc['lambda_normal_mu', 'Mean'])
print('  SD:', summary0.loc['lambda_normal_sigma', 'StdDev'])
print()
print('Posterior of mu for lambda = 1:')
print('  Mean:', summary1.loc['lambda_normal_mu', 'Mean'])
print('  SD:', summary1.loc['lambda_normal_sigma', 'StdDev'])

#%%
ln1 = get_lambdas(summaryn1)
l0 = get_lambdas(summary0)
l1 = get_lambdas(summary1)
xn1 = np.full_like(ln1, -1)
x0 = np.full_like(l0, 0)
x1 = np.full_like(l1, 1)
x = sm.add_constant(np.concatenate((xn1, x0, x1)))
y = np.concatenate((ln1, l0, l1))
results = sm.OLS(y, x).fit()

#%%
print('OLS regression on mean of posterior over \hat\lambda:\n')
print(results.summary())
