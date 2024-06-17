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
 nt_x_fix_cols, nt_x_rnd_cols) = Ddm().generate_data(N=200,T=5)

df.to_pickle('ddm_data.pickle')