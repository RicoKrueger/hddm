import numpy as np
import pandas as pd
from dcddm import Ddm

np.random.seed(124)

###
#Generate data
###
(df, 
 mu_x_fix_cols, mu_x_rnd_cols, 
 w_x_fix_cols, w_x_rnd_cols,
 a_x_fix_cols, a_x_rnd_cols,
 nt_x_fix_cols, nt_x_rnd_cols) = Ddm().generate_data(N=100,T=10)

print(df)

###
#Estimate model
###

results = Ddm().estimate(
    df=df, 
    mu_x_fix_cols=mu_x_fix_cols, 
    mu_x_rnd_cols=mu_x_rnd_cols, 
    w_x_fix_cols=w_x_fix_cols, 
    w_x_rnd_cols=w_x_rnd_cols, 
    a_x_fix_cols=a_x_fix_cols, 
    a_x_rnd_cols=None, 
    n_draws=100, 
    compute_hess=True
    )

###
#Elasticities
###

t_steps = np.arange(1, 15+1)
probs_old = Ddm().predict(
    results,
    df, 
    ind_id_col='ind_id', 
    obs_id_col='obs_id', 
    y_col='y',
    mu_x_fix_cols=mu_x_fix_cols, 
    mu_x_rnd_cols=mu_x_rnd_cols, 
    w_x_fix_cols=w_x_fix_cols, 
    w_x_rnd_cols=w_x_rnd_cols, 
    a_x_fix_cols=a_x_fix_cols, 
    a_x_rnd_cols=None, 
    n_draws=100, 
    predict_loglik=False, t_steps=t_steps
    )

print(pd.DataFrame(probs_old.reshape(-1,t_steps.shape[0]), columns=t_steps))

def arc_elasticity(df, x_name, delta, probs_old):
    df_new = df.copy()
    df_new[x_name] *= (1 + delta)

    probs_new = Ddm().predict(
        results,
        df_new, 
        ind_id_col='ind_id', 
        obs_id_col='obs_id', 
        y_col='y',
        mu_x_fix_cols=mu_x_fix_cols, 
        mu_x_rnd_cols=mu_x_rnd_cols, 
        w_x_fix_cols=w_x_fix_cols, 
        w_x_rnd_cols=w_x_rnd_cols, 
        a_x_fix_cols=a_x_fix_cols, 
        a_x_rnd_cols=None, 
        n_draws=100, 
        predict_loglik=False, t_steps=t_steps
        ) # n_ind, n_sit, n_steps
    
    elas_numer = (probs_new - probs_old) / ((probs_new + probs_old) / 2)
    x_old = df[x_name].values.reshape(probs_old.shape[0], probs_old.shape[1], 1)
    x_new = df_new[x_name].values.reshape(probs_old.shape[0], probs_old.shape[1], 1)
    elas_denom = (x_new - x_old) / ((x_new + x_old) / 2)
    elas = elas_numer / elas_denom

    return elas, probs_new

delta = 0.05
elas, probs_new = arc_elasticity(df, 'x2', delta, probs_old)
elas_mean = elas.mean(axis=(0,1))
print(elas_mean)
#print(pd.DataFrame(elas.reshape(-1,t_steps.shape[0]), columns=t_steps))
#print(pd.DataFrame(probs_new.reshape(-1,t_steps.shape[0]), columns=t_steps))

def pseudo_elasticity(df, x_name, level_base, level_new):
    df_base = df.copy()
    df_base[x_name] = level_base

    probs_base = Ddm().predict(
        results,
        df_base, 
        ind_id_col='ind_id', 
        obs_id_col='obs_id', 
        y_col='y',
        mu_x_fix_cols=mu_x_fix_cols, 
        mu_x_rnd_cols=mu_x_rnd_cols, 
        w_x_fix_cols=w_x_fix_cols, 
        w_x_rnd_cols=w_x_rnd_cols, 
        a_x_fix_cols=a_x_fix_cols, 
        a_x_rnd_cols=None, 
        n_draws=100, 
        predict_loglik=False, t_steps=t_steps
        ) # n_ind, n_sit, n_steps

    df_new = df.copy()
    df_new[x_name] = level_new

    probs_new = Ddm().predict(
        results,
        df_new, 
        ind_id_col='ind_id', 
        obs_id_col='obs_id', 
        y_col='y',
        mu_x_fix_cols=mu_x_fix_cols, 
        mu_x_rnd_cols=mu_x_rnd_cols, 
        w_x_fix_cols=w_x_fix_cols, 
        w_x_rnd_cols=w_x_rnd_cols, 
        a_x_fix_cols=a_x_fix_cols, 
        a_x_rnd_cols=None, 
        n_draws=100, 
        predict_loglik=False, t_steps=t_steps
        ) # n_ind, n_sit, n_steps
    
    elas = (probs_new - probs_base) / probs_base

    return elas
    
level_base = 0.0 
level_new = 1.0
elas = pseudo_elasticity(df, 'x2', level_base, level_new)
elas_mean = elas.mean(axis=(0,1))
print(elas_mean)
#print(pd.DataFrame(elas.reshape(-1,t_steps.shape[0]), columns=t_steps))
#print(pd.DataFrame(probs_new.reshape(-1,t_steps.shape[0]), columns=t_steps))
