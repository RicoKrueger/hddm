import numpy as np
import pandas as pd
from dcddm import Logit

np.random.seed(124)

###
#Generate data
###

df, x_fix_cols, x_rnd_cols = Logit()\
    .generate_data()

###
#Estimate model
###

results = Logit().estimate(
    df=df, 
    x_fix_cols=x_fix_cols,
    x_rnd_cols=x_rnd_cols,
    n_draws=200, compute_hess=True
    )

###
#Elasticities
###

n_alt = 3

probs_old = Logit().predict(
    results,
    df, 
    x_fix_cols=x_fix_cols,
    x_rnd_cols=x_rnd_cols,
    n_draws=200, 
    predict_loglik=False
    ) #n_ind x n_sit x n_alt

print(pd.DataFrame(probs_old.reshape(-1, n_alt)))

def arc_elasticity(df, x_name, delta, probs_old, x_alt_id, prob_alt_id):
    df_new = df.copy()
    x_mask = df['alt_id'] == x_alt_id
    df_new[x_name] = np.where(x_mask, df_new[x_name] * (1 + delta), df[x_name])

    probs_new = Logit().predict(
        results,
        df_new, 
        x_fix_cols=x_fix_cols,
        x_rnd_cols=x_rnd_cols,
        n_draws=200, 
        predict_loglik=False
        ) #n_ind x n_sit x n_alt
    
    p_mask = (df['alt_id'].values == prob_alt_id).reshape(probs_old.shape)
    p_old = probs_old[p_mask].copy().reshape(probs_old.shape[0], probs_old.shape[1])
    p_new = probs_new[p_mask].copy().reshape(probs_new.shape[0], probs_new.shape[1])
    elas_numer = (p_new - p_old) / ((p_new + p_old) / 2) #n_ind x n_sit

    x_old = df[x_name][x_mask].values.reshape(probs_old.shape[0], probs_old.shape[1])
    x_new = df_new[x_name][x_mask].values.reshape(probs_new.shape[0], probs_new.shape[1])
    elas_denom = (x_new - x_old) / ((x_new + x_old) / 2)

    elas = elas_numer / elas_denom

    return elas

delta = 0.05
elas = arc_elasticity(df, 'asc1', delta, probs_old, 1, 1)
elas_mean = elas.mean()
print(elas_mean)
#print(pd.DataFrame(elas.reshape(probs_old.shape[0], probs_old.shape[1])))
    
def pseudo_elasticity(df, x_name, x_alt_id, level_base, level_new, prob_alt_id):
    df_base = df.copy()
    x_mask = df['alt_id'] == x_alt_id
    df_base[x_name] = np.where(x_mask, level_base, df[x_name])

    probs_base = Logit().predict(
        results,
        df_base, 
        x_fix_cols=x_fix_cols,
        x_rnd_cols=x_rnd_cols,
        n_draws=200, 
        predict_loglik=False
        ) #n_ind x n_sit x n_alt

    df_new = df.copy()
    df_new[x_name] = np.where(x_mask, level_new, df[x_name])

    probs_new = Logit().predict(
        results,
        df_new, 
        x_fix_cols=x_fix_cols,
        x_rnd_cols=x_rnd_cols,
        n_draws=200, 
        predict_loglik=False
        ) #n_ind x n_sit x n_alt
    
    p_mask = (df['alt_id'].values == prob_alt_id).reshape(probs_base.shape)
    p_base = probs_base[p_mask].copy().reshape(probs_base.shape[0], probs_base.shape[1])
    p_new = probs_new[p_mask].copy().reshape(probs_new.shape[0], probs_new.shape[1])

    elas = (p_new - p_base) / p_base

    return elas

level_base = 0.0 
level_new = 1.0
elas = pseudo_elasticity(df, 'asc1', 1, level_base, level_new, 1)
elas_mean = elas.mean()
print(elas_mean)
#print(pd.DataFrame(elas.reshape(probs_old.shape[0], probs_old.shape[1])))