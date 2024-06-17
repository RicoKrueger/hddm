import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.optimize import minimize
import itertools as it
#import numdifftools as nd
from findiff import hessian
import datetime
from ddm_functions import *
     
class Logit:
    
    def __reshape_x(self, x_in):
        """ Reshapes explanatory variables from a data frame to an array of size
        self.n_ind X 1 X self.n_sit X self.n_alt X -1 """
        
        x = x_in.values.reshape(self.n_ind, 1, self.n_sit, self.n_alt, -1)
        return x
    
        
    def __get_parameters(self, param_arr):
        """ Gets the parameters from a 1-d array """
        
        i = 0
        j = 0
        if self.n_fix:
            i = j
            j += self.n_fix
            self.__beta = np.array(param_arr[i:j])
        if self.n_rnd:
            i = j
            j += self.n_rnd
            self.__sigma = np.array(param_arr[i:j])
            self.__xi = self.__sigma.reshape(1,1,self.n_rnd) * self.draws
             
    
    def __calc_v(self):
        """ Calculates the deterministic utility component. """
        
        v = np.zeros((self.n_ind,self.n_draws,self.n_sit,self.n_alt))
        if self.n_fix:
            v += np.sum(
                self.x_fix * self.__beta.reshape(1,1,1,1,self.n_fix), 
                axis=4
                )
        if self.n_rnd:
            v += np.sum(
                self.x_rnd * self.__xi.reshape(
                    self.n_ind,self.n_draws,1,1,self.n_rnd
                    ), 
                axis=4)
            
        return v
    
    def __objective_loglik(self, param_arr):
        """ Computes the negative log-likelihood """
        
        self.__get_parameters(param_arr)
        
        v = self.__calc_v()
        v_chosen = np.sum(v * self.chosen, axis=3)
        log_denom = logsumexp(v, axis=3)
        lp_chosen_sim = v_chosen - log_denom
        lp_ind_sim = lp_chosen_sim.sum(axis=2)
        p_ind_sim = np.exp(lp_ind_sim)
        p_ind = p_ind_sim.mean(axis=1)
        p_ind[p_ind < 1e-300] = 1e-300
        lp_ind = np.log(p_ind)
        ll = np.sum(lp_ind)
        neg_ll = -ll
        
        return neg_ll
    
    def __predict_probs(self, param_arr):
        """ Computes predicted probabilities """
        
        self.__get_parameters(param_arr)
        
        v = self.__calc_v() #n_ind x n_draws x n_sit x n_alt
        ev = np.exp(v)
        denom = ev.sum(axis=3, keepdims=True)
        probs = ev / denom
        probs = probs.mean(axis=1) #n_ind x n_sit x n_alt
        
        return probs
        
    
    def _mlhs(self):
        draws = np.zeros((self.n_ind, self.n_draws, self.n_rnd))
        h = np.arange(self.n_draws)
        for n, i in it.product(np.arange(self.n_ind), np.arange(self.n_rnd)):
            d = h + np.random.rand()
            draws[n,:,i] = np.random.choice(d, size=self.n_draws, replace=False)
        draws /= self.n_draws
        draws = norm.ppf(draws)
        return draws
    
    def _gen_primes(self, n):
        """ Generate an infinite sequence of prime numbers.
        """
        """
        Source: https://stackoverflow.com/questions/567222/simple-prime-\
            number-generator-in-python
        """
        # Maps composites to primes witnessing their compositeness.
        # This is memory efficient, as the sieve is not "run forward"
        # indefinitely, but only as long as required by the current
        # number being tested.
        #
        D = {}
        # The running integer that's checked for primeness
        q = 2
        i = 0
        while True:
            if q not in D:
                # q is a new prime.
                # Yield it and mark its first multiple that isn't
                # already marked in previous iterations
                # 
                i += 1
                yield q
                if i >= n:
                    break
                D[q * q] = [q]
            else:
                # q is composite. D[q] is the list of primes that
                # divide it. Since we've reached q, we no longer
                # need it in the map, but we'll mark the next 
                # multiples of its witnesses to prepare for larger
                # numbers
                # 
                for p in D[q]:
                    D.setdefault(p + q, []).append(p)
                del D[q]
            
            q += 1
            
    def _get_primes(self, n):
        return list(self._gen_primes(n))
    
    def _gen_halton(self, n_seq, base=2, skip=10):
        n, d = 0, 1
        seq = np.empty((n_seq,))
        for i in range(n_seq + skip):
            x = d - n
            if x == 1:
                n = 1
                d *= base
            else:
                y = d // base
                while x <= y:
                    y //= base
                n = (base + 1) * y - x
            if i >= skip:
                seq[i - skip] = n / d
        return seq
    
    def _halton(self):
        draws = np.zeros((self.n_ind, self.n_draws, self.n_rnd))
        bases = self._get_primes(self.n_rnd)
        n_seq = int(self.n_ind * self.n_draws)
        for i, b in enumerate(bases):
            #Generate Halton sequence
            d = self._gen_halton(n_seq, b)
            d = d.reshape(self.n_ind, self.n_draws)
            d += np.random.rand()
            d -= np.floor(d)
            #Shuffle and store
            for n, d_n in enumerate(d):
                draws[n,:,i] = np.random.choice(
                    d_n, size=self.n_draws, replace=False
                    )
        draws = norm.ppf(draws)
        return draws
  
    def __initialise(
            self,
            df, 
            ind_id_col,
            obs_id_col,
            alt_id_col,
            chosen_col,
            x_fix_cols, x_rnd_cols,
            n_draws
            ):
        
        self.ind_id_col = ind_id_col
        self.obs_id_col = obs_id_col
        self.alt_id_col = alt_id_col
        self.chosen_col = chosen_col
        
        self.n_ind = df[self.ind_id_col].unique().shape[0]
        self.n_obs = df[self.obs_id_col].unique().shape[0]
        self.n_sit = int(self.n_obs / self.n_ind)
        self.n_alt = df[self.alt_id_col].max()
    
        self.chosen = df[self.chosen_col].values.reshape(
            self.n_ind, 1, self.n_sit, self.n_alt
            )
        
        self.x_fix_cols = x_fix_cols
        if self.x_fix_cols is not None:
            self.n_fix = len(self.x_fix_cols)
            self.x_fix = self.__reshape_x(df[self.x_fix_cols])
        else:
            self.n_fix = 0
        
        self.x_rnd_cols = x_rnd_cols
        if self.x_rnd_cols is not None:
            self.n_rnd = len(self.x_rnd_cols)
            self.x_rnd = self.__reshape_x(df[self.x_rnd_cols])
        else:
            self.n_rnd = 0
        
        self.draws = None
        
        if self.n_rnd:
            self.n_draws = n_draws
            self.draws = self._mlhs()
            #self.draws = self._halton()
        else:
            self.n_draws = 1
    
    def estimate(
            self, 
            df, 
            ind_id_col='ind_id', 
            obs_id_col='obs_id', 
            alt_id_col='alt_id',
            chosen_col='chosen',
            x_fix_cols=None,
            x_rnd_cols=None,
            inits=None, n_draws=1,
            compute_hess=False
            ):
        
        ###
        #Initialise
        ###
        
        self.__initialise(
            df, 
            ind_id_col,
            obs_id_col,
            alt_id_col,
            chosen_col,
            x_fix_cols,
            x_rnd_cols,
            n_draws
            )  
            
        ###
        #Initial values
        ###
        
        if inits is None:
            self.inits = np.concatenate((
                np.zeros((self.n_fix,)),
                np.ones((self.n_rnd,))
                ))
        else:
            self.inits = inits
        
            
        ###
        #Maximum likelihood estimation
        ###
        
        results = {}
        
        time_start = datetime.datetime.now()
        results['time_start'] = time_start.strftime('%Y-%m-%d %H:%M:%S')
        print(' ')
        print('Estimation started at', results['time_start'])
        
        #grad_fun = grad(self.__objective_loglik)

        res = minimize(
            fun=self.__objective_loglik, 
            x0=self.inits, 
            method='L-BFGS-B',
            jac=False,
            options={
                'disp': True,
                'gtol': 1e-05
                }
            )
            
        results['res'] = res

        #Standard errors        
        est = res['x']
        
        if compute_hess:
            print('Computing Hessian using finite differences.')
            #hess_fun = nd.Hessian(self.__objective_loglik)
            #hess = hess_fun(est)
            hess = hessian(self.__objective_loglik, est)
            iHess = np.linalg.inv(hess)
            se = np.sqrt(np.diag(iHess))
            z_val = est / se
            p_val = 2 * norm.cdf(-np.absolute(z_val))
            
        time_end = datetime.datetime.now()
        results['time_end'] = time_end.strftime('%Y-%m-%d %H:%M:%S')
        print('Estimation completed at', results['time_end'])
        time_dur = (time_end - time_start).total_seconds()
        results['estimation_time'] = time_dur
        print('Estimation time [s]:', results['estimation_time'])
        
        ###
        #Extract results
        ###
            
        results['est'] = est
        results['n_param'] = est.shape[0]
        results['loglik'] = -res['fun']
        results['aic'] = 2 * results['n_param'] - 2 * results['loglik']
        
        print('No. of parameters:', results['n_param'])
        print('Final log-lik.:', results['loglik'])
        print('AIC:', results['aic'])
    
        param_names = []
        if self.n_fix:
            param_names.extend(self.x_fix_cols)
            results['x_fix_cols'] = self.x_fix_cols
        if self.n_rnd:
            param_rnd_names = [f'sigma_{i}' for i in self.x_rnd_cols]
            results['param_rnd_names'] = param_rnd_names
            results['x_rnd_cols'] = self.x_rnd_cols
            param_names.extend(param_rnd_names)

        results['estimates'] = pd.DataFrame(
            data={'Est.': est},
            index=param_names
            )
        if compute_hess:
            results['estimates']['SE'] = se
            results['estimates']['z-val.'] = z_val
            results['estimates']['p-val.'] = p_val
            z = norm.ppf(0.975)
            results['estimates']['[2.5%'] = est - z * se
            results['estimates']['97.5%]'] = est + z * se
        
        print(' ')
        print('Estimates:')
        print(results['estimates'])
        
        return results
    
    def predict(
            self, 
            results,
            df, 
            ind_id_col='ind_id', 
            obs_id_col='obs_id', 
            alt_id_col='alt_id',
            chosen_col='chosen',
            x_fix_cols=None,
            x_rnd_cols=None,
            n_draws=1,
            predict_loglik=True
            ):
        
        ###
        #Initialise
        ###
        
        self.__initialise(
            df, 
            ind_id_col,
            obs_id_col,
            alt_id_col,
            chosen_col,
            x_fix_cols,
            x_rnd_cols,
            n_draws
            )  

        if predict_loglik:    

            ###
            #Predict log-likelihood
            ###
            
            loglik = -self.__objective_loglik(results['est'])
            print('Predictive log-lik.', loglik)
            
            return loglik
        
        else:

            ###
            #Predict probs
            ###

            probs = self.__predict_probs(results['est'])

            return probs

    def generate_data(self, N=1000, T=10):
        
        print(' ')
        print('Generating synthetic data for logit.')
        
        ###
        #Initialise and generate data
        ###
        
        self.n_ind = N
        self.n_obs = N * T
        self.n_sit = T
        self.n_alt = 3
        
        df = pd.DataFrame()
        df['ind_id'] = np.repeat(
            np.arange(1, self.n_ind + 1), self.n_sit * self.n_alt
            )
        df['obs_id'] = np.repeat(np.arange(1, self.n_obs + 1), self.n_alt)
        df['alt_id'] = np.tile(np.arange(1, self.n_alt + 1), self.n_obs)
        
        self.x_fix_cols = ['asc1', 'asc2']
        df['asc1'] = np.where(df['alt_id']==1, 1, 0)
        df['asc2'] = np.where(df['alt_id']==2, 1, 0)
        
        self.x_rnd_cols = ['1', '2']
        df['1'] = np.where(df['alt_id']==1, 1, 0)
        df['2'] = np.where(df['alt_id']==2, 1, 0)
        
        self.n_fix = len(self.x_fix_cols)
        self.x_fix = self.__reshape_x(df[self.x_fix_cols])
        
        self.n_rnd = len(self.x_rnd_cols)
        self.x_rnd = self.__reshape_x(df[self.x_rnd_cols])
        self.n_draws = 1
        self.draws = np.random.randn(self.n_ind, self.n_draws, self.n_rnd)
        
        ###
        #Set true parameter values
        ###
        
        self.__beta = np.array([1, -1])
        
        self.__sigma = np.array([1, 1.5])
        self.__xi = self.__sigma.reshape(1,1,self.n_rnd) * self.draws
        
        ###
        #Generate choices
        ###
        
        v = self.__calc_v().reshape((self.n_ind, self.n_sit, self.n_alt))
        v_max = np.argmax(v, axis=2)
        eps = -np.log(-np.log(np.random.rand(self.n_ind, self.n_sit, self.n_alt)))
        u = v + eps
        u_max = np.argmax(u, axis=2)
        
        error_rate = np.mean(u_max != v_max)
        print('Error rate:', error_rate)
    
        chosen = np.zeros((self.n_ind, self.n_sit, self.n_alt))
        for i, j in it.product(np.arange(self.n_ind), np.arange(self.n_sit)):
            k = u_max[i,j]
            chosen[i,j,k] = 1
        chosen = chosen.reshape(-1,)
        df.insert(loc=3, column='chosen', value=chosen)
        
        return (
            df, 
            self.x_fix_cols, self.x_rnd_cols
            )
        
class Ddm(Logit):

    def __reshape_x(self, x_in):
        """ Reshapes explanatory variables from a data frame to an array of size
        self.n_ind X 1 X self.n_sit X -1 """
        
        x = x_in.values.reshape(self.n_ind, 1, self.n_sit, -1)
        return x

    def __get_parameters(self, param_arr):
        """ Gets the parameters from a 1-d array """
        # DDM parameters are mu, w, a, nt
        
        i = 0
        j = 0
        ii = 0
        jj = 0

        #mu
        if self.mu_n_fix:
            i = j
            j += self.mu_n_fix
            self.__mu_beta = np.array(param_arr[i:j])
        if self.mu_n_rnd:
            i = j
            j += self.mu_n_rnd
            ii = jj
            jj += self.mu_n_rnd
            self.__mu_sigma = np.array(param_arr[i:j])
            self.__mu_xi = self.__mu_sigma.reshape(1,1,self.mu_n_rnd) * self.draws[:,:,ii:jj]

        #w
        if self.w_n_fix:
            i = j
            j += self.w_n_fix
            self.__w_beta = np.array(param_arr[i:j])
        if self.w_n_rnd:
            i = j
            j += self.w_n_rnd
            ii = jj
            jj += self.w_n_rnd
            self.__w_sigma = np.array(param_arr[i:j])
            self.__w_xi = self.__w_sigma.reshape(1,1,self.w_n_rnd) * self.draws[:,:,ii:jj]  

        #a
        if self.a_n_fix:
            i = j
            j += self.a_n_fix
            self.__a_beta = np.array(param_arr[i:j])
        if self.a_n_rnd:
            i = j
            j += self.a_n_rnd
            ii = jj
            jj += self.a_n_rnd
            self.__a_sigma = np.array(param_arr[i:j])
            self.__a_xi = self.__a_sigma.reshape(1,1,self.a_n_rnd) * self.draws[:,:,ii:jj]

        #nt
        if self.nt_n_fix:
            i = j
            j += self.nt_n_fix
            self.__nt_beta = np.array(param_arr[i:j])
        if self.nt_n_rnd:
            i = j
            j += self.nt_n_rnd
            ii = jj
            jj += self.nt_n_rnd
            self.__nt_sigma = np.array(param_arr[i:j])
            self.__nt_xi = self.__nt_sigma.reshape(1,1,self.nt_n_rnd) * self.draws[:,:,ii:jj]                       
    
    def __calc_mu(self):
        """ Calculates drift rate parameter mu. """
        mu = np.zeros((self.n_ind, self.n_draws, self.n_sit))
        if self.mu_n_fix:
            mu += np.sum(
                self.mu_x_fix * self.__mu_beta.reshape(1,1,1,self.mu_n_fix), 
                axis=3
                )
        if self.mu_n_rnd:
            mu += np.sum(
                self.mu_x_rnd * self.__mu_xi.reshape(
                    self.n_ind,self.n_draws,1,self.mu_n_rnd
                    ), 
                axis=3)  
        return mu # n_ind x n_draws x n_sit
    
    def __calc_w(self):
        """ Calculates bias parameter w. """ 
        w = np.zeros((self.n_ind, self.n_draws, self.n_sit))
        if self.w_n_fix:
            w += np.sum(
                self.w_x_fix * self.__w_beta.reshape(1,1,1,self.w_n_fix), 
                axis=3
                )
        if self.w_n_rnd:
            w += np.sum(
                self.w_x_rnd * self.__w_xi.reshape(
                    self.n_ind,self.n_draws,1,self.w_n_rnd
                    ), 
                axis=3)
        
        #w[w > 1e2] = 1e2 
        #w[w < -1e2] = -1e2
        w = 1 / (1 + np.exp(-w))
        return w # n_ind x n_draws x n_sit
    
    def __calc_a(self):
        """ Calculates threshold parameter a. """
        a = np.zeros((self.n_ind, self.n_draws, self.n_sit))
        if self.a_n_fix:
            a += np.sum(
                self.a_x_fix * self.__a_beta.reshape(1,1,1,self.a_n_fix), 
                axis=3
                )
        if self.a_n_rnd:
            a += np.sum(
                self.a_x_rnd * self.__a_xi.reshape(
                    self.n_ind,self.n_draws,1,self.a_n_rnd
                    ), 
                axis=3)
        
        #a[a > 1e2] = 1e2 
        a = np.exp(a)
        return a # n_ind x n_draws x n_sit

    def __calc_nt(self):
        """ Calculates the non-decision time nt. """
        nt = np.zeros((self.n_ind, self.n_draws, self.n_sit))
        if self.nt_n_fix:
            nt += np.sum(
                self.nt_x_fix * self.__nt_beta.reshape(1,1,1,self.nt_n_fix), 
                axis=3
                )
        if self.nt_n_rnd:
            nt += np.sum(
                self.nt_x_rnd * self.__nt_xi.reshape(
                    self.n_ind,self.n_draws,1,self.t_n_rnd
                    ), 
                axis=3)
            
        if self.nt_n_fix > 0 or self.nt_n_rnd > 0:
            nt = 1 / (1 + np.exp(-nt))
            nt *= 1
        return nt # n_ind x n_draws x n_sit
    
    def __objective_loglik(self, param_arr):
        """ Computes the negative log-likelihood """
        
        self.__get_parameters(param_arr)
        
        mu = self.__calc_mu()
        w = self.__calc_w()
        a = self.__calc_a()
        nt = self.__calc_nt()

        lp_decision_sim = ddm_lp(
            self.y, mu, w, a, nt, 
            self.n_ind, self.n_draws, self.n_sit
            ) #n_ind x n_draws x n_sit

        lp_ind_sim = lp_decision_sim.sum(axis=2)
        lp_ind_sim[lp_ind_sim > 1e2] = 1e2
        p_ind_sim = np.exp(lp_ind_sim)
        p_ind = p_ind_sim.mean(axis=1)
        p_ind[p_ind < 1e-300] = 1e-300
        lp_ind = np.log(p_ind)
        ll = np.sum(lp_ind)
        neg_ll = -ll
        
        return neg_ll  
    
    def __predict_probs(self, param_arr, t_steps):
        """ Computes the predicted probabilities across several time steps """
        
        self.__get_parameters(param_arr)
        
        mu = self.__calc_mu()
        w = self.__calc_w()
        a = self.__calc_a()
        nt = self.__calc_nt()

        probs_sim = ddm_cdf(
            self.y, mu, w, a, nt, 
            self.n_ind, self.n_draws, self.n_sit,
            t_steps
            ) #n_ind x n_draws x n_sit x n_steps
        probs = probs_sim.mean(axis=1) #n_ind x n_sit x n_steps

        return probs
    
    def __predict_rt(self, param_arr):
        """ Computes the expected response times for absorption at the upper boundary """
        
        self.__get_parameters(param_arr)
        
        mu = self.__calc_mu() # n_ind x n_draws x n_sit
        w = self.__calc_w()
        a = self.__calc_a()
        nt = self.__calc_nt()

        rt_sim = ddm_rt_upper(
            mu, w, a, nt, 
            self.n_ind, self.n_draws, self.n_sit
            ) #n_ind x n_draws x n_sit
        rt = rt_sim.mean(axis=1) #n_ind x n_sit
        return rt

    def __initialise(
            self,
            df, 
            ind_id_col,
            obs_id_col,
            y_col,
            mu_x_fix_cols, mu_x_rnd_cols,
            w_x_fix_cols, w_x_rnd_cols,
            a_x_fix_cols, a_x_rnd_cols,
            nt_x_fix_cols, nt_x_rnd_cols,
            n_draws
            ):
        
        self.ind_id_col = ind_id_col
        self.obs_id_col = obs_id_col
        self.y_col = y_col
        
        self.n_ind = df[self.ind_id_col].unique().shape[0]
        self.n_obs = df[self.obs_id_col].unique().shape[0]
        self.n_sit = int(self.n_obs / self.n_ind)
    
        self.y = df[self.y_col].values.reshape(self.n_ind, self.n_sit)
        
        #mu
        self.mu_x_fix_cols = mu_x_fix_cols
        if self.mu_x_fix_cols is not None:
            self.mu_n_fix = len(self.mu_x_fix_cols)
            self.mu_x_fix = self.__reshape_x(df[self.mu_x_fix_cols])
        else:
            self.mu_n_fix = 0
        
        self.mu_x_rnd_cols = mu_x_rnd_cols
        if self.mu_x_rnd_cols is not None:
            self.mu_n_rnd = len(self.mu_x_rnd_cols)
            self.mu_x_rnd = self.__reshape_x(df[self.mu_x_rnd_cols])
        else:
            self.mu_n_rnd = 0

        #w
        self.w_x_fix_cols = w_x_fix_cols
        if self.w_x_fix_cols is not None:
            self.w_n_fix = len(self.w_x_fix_cols)
            self.w_x_fix = self.__reshape_x(df[self.w_x_fix_cols])
        else:
            self.w_n_fix = 0
        
        self.w_x_rnd_cols = w_x_rnd_cols
        if self.w_x_rnd_cols is not None:
            self.w_n_rnd = len(self.w_x_rnd_cols)
            self.w_x_rnd = self.__reshape_x(df[self.w_x_rnd_cols])
        else:
            self.w_n_rnd = 0     

        #a
        self.a_x_fix_cols = a_x_fix_cols
        if self.a_x_fix_cols is not None:
            self.a_n_fix = len(self.a_x_fix_cols)
            self.a_x_fix = self.__reshape_x(df[self.a_x_fix_cols])
        else:
            self.a_n_fix = 0
        
        self.a_x_rnd_cols = a_x_rnd_cols
        if self.a_x_rnd_cols is not None:
            self.a_n_rnd = len(self.a_x_rnd_cols)
            self.a_x_rnd = self.__reshape_x(df[self.a_x_rnd_cols])
        else:
            self.a_n_rnd = 0  

        #nt
        self.nt_x_fix_cols = nt_x_fix_cols
        if self.nt_x_fix_cols is not None:
            self.nt_n_fix = len(self.nt_x_fix_cols)
            self.nt_x_fix = self.__reshape_x(df[self.nt_x_fix_cols])
        else:
            self.nt_n_fix = 0
        
        self.nt_x_rnd_cols = nt_x_rnd_cols
        if self.nt_x_rnd_cols is not None:
            self.nt_n_rnd = len(self.nt_x_rnd_cols)
            self.nt_x_rnd = self.__reshape_x(df[self.nt_x_rnd_cols])
        else:
            self.nt_n_rnd = 0        
        

        self.n_rnd = self.mu_n_rnd + self.w_n_rnd + self.a_n_rnd + self.nt_n_rnd
        self.draws = None
        
        if self.n_rnd:
            self.n_draws = n_draws
            #self.draws = self._mlhs()
            self.draws = self._halton()
        else:
            self.n_draws = 1
    
    def estimate(
            self, 
            df, 
            ind_id_col='ind_id', 
            obs_id_col='obs_id', 
            y_col='y',
            mu_x_fix_cols=None, mu_x_rnd_cols=None,
            w_x_fix_cols=None, w_x_rnd_cols=None,
            a_x_fix_cols=None, a_x_rnd_cols=None,
            nt_x_fix_cols=None, nt_x_rnd_cols=None,
            inits=None, n_draws=1,
            compute_hess=False
            ):
        
        ###
        #Initialise
        ###
        
        self.__initialise(
            df, 
            ind_id_col,
            obs_id_col,
            y_col,
            mu_x_fix_cols, mu_x_rnd_cols,
            w_x_fix_cols, w_x_rnd_cols,
            a_x_fix_cols, a_x_rnd_cols,
            nt_x_fix_cols, nt_x_rnd_cols,
            n_draws
            )  
            
        ###
        #Initial values
        ###
        
        if inits is None:
            self.inits = np.concatenate((
                np.zeros((self.mu_n_fix,)), 0.1 * np.ones((self.mu_n_rnd,)),
                np.zeros((self.w_n_fix,)), 0.1 * np.ones((self.w_n_rnd,)),
                np.zeros((self.a_n_fix,)), 0.1 * np.ones((self.a_n_rnd,)),
                np.zeros((self.nt_n_fix,)), 0.1 * np.ones((self.nt_n_rnd,))
                ))
        else:
            self.inits = inits
        
            
        ###
        #Maximum likelihood estimation
        ###
        
        results = {}
        
        time_start = datetime.datetime.now()
        results['time_start'] = time_start.strftime('%Y-%m-%d %H:%M:%S')
        print(' ')
        print('Estimation started at', results['time_start'])
        
        #grad_fun = grad(self.__objective_loglik)

        res = minimize(
            fun=self.__objective_loglik, 
            x0=self.inits, 
            method='L-BFGS-B',
            jac=False, #grad_fun,
            options={
                'disp': True,
                'gtol': 1e-06
                }
            )
            
        results['res'] = res

        #Standard errors        
        est = res['x']
        
        if compute_hess:
            print('Computing Hessian using finite differences.')
            #hess_fun = nd.Hessian(self.__objective_loglik)
            #hess = hess_fun(est)
            hess = hessian(self.__objective_loglik, est)
            iHess = np.linalg.inv(hess)
            se = np.sqrt(np.diag(iHess))
            z_val = est / se
            p_val = 2 * norm.cdf(-np.absolute(z_val))
            
        time_end = datetime.datetime.now()
        results['time_end'] = time_end.strftime('%Y-%m-%d %H:%M:%S')
        print('Estimation completed at', results['time_end'])
        time_dur = (time_end - time_start).total_seconds()
        results['estimation_time'] = time_dur
        print('Estimation time [s]:', results['estimation_time'])
        
        ###
        #Extract results
        ###
            
        results['est'] = est
        results['n_param'] = est.shape[0]
        results['loglik'] = -res['fun']
        results['aic'] = 2 * results['n_param'] - 2 * results['loglik']
        
        print('No. of parameters:', results['n_param'])
        print('Final log-lik.:', results['loglik'])
        print('AIC:', results['aic'])
    
        param_names = []

        #mu
        if self.mu_n_fix:
            param_fix_names = [f'mu_{i}' for i in self.mu_x_fix_cols]
            param_names.extend(param_fix_names)
            results['mu_x_fix_cols'] = self.mu_x_fix_cols
        if self.mu_n_rnd:
            param_rnd_names = [f'mu_sigma_{i}' for i in self.mu_x_rnd_cols]
            results['mu_param_rnd_names'] = param_rnd_names
            results['mu_x_rnd_cols'] = self.mu_x_rnd_cols
            param_names.extend(param_rnd_names)

        #w
        if self.w_n_fix:
            param_fix_names = [f'w_{i}' for i in self.w_x_fix_cols]
            param_names.extend(param_fix_names)
            results['w_x_fix_cols'] = self.w_x_fix_cols
        if self.w_n_rnd:
            param_rnd_names = [f'w_sigma_{i}' for i in self.w_x_rnd_cols]
            results['w_param_rnd_names'] = param_rnd_names
            results['w_x_rnd_cols'] = self.w_x_rnd_cols
            param_names.extend(param_rnd_names)

        #a
        if self.a_n_fix:
            param_fix_names = [f'a_{i}' for i in self.a_x_fix_cols]
            param_names.extend(param_fix_names)
            results['a_x_fix_cols'] = self.a_x_fix_cols
        if self.a_n_rnd:
            param_rnd_names = [f'a_sigma_{i}' for i in self.a_x_rnd_cols]
            results['a_param_rnd_names'] = param_rnd_names
            results['a_x_rnd_cols'] = self.a_x_rnd_cols
            param_names.extend(param_rnd_names)

        #nt
        if self.nt_n_fix:
            param_fix_names = [f'nt_{i}' for i in self.nt_x_fix_cols]
            param_names.extend(param_fix_names)
            results['nt_x_fix_cols'] = self.nt_x_fix_cols
        if self.nt_n_rnd:
            param_rnd_names = [f'nt_sigma_{i}' for i in self.nt_x_rnd_cols]
            results['nt_param_rnd_names'] = param_rnd_names
            results['nt_x_rnd_cols'] = self.nt_x_rnd_cols
            param_names.extend(param_rnd_names)

        results['estimates'] = pd.DataFrame(
            data={'Est.': est},
            index=param_names
            )
        if compute_hess:
            results['estimates']['SE'] = se
            results['estimates']['z-val.'] = z_val
            results['estimates']['p-val.'] = p_val
            z = norm.ppf(0.975)
            results['estimates']['[2.5%'] = est - z * se
            results['estimates']['97.5%]'] = est + z * se
        
        print(' ')
        print('Estimates:')
        print(results['estimates'])
        
        return results
    
    def predict(
            self, 
            results,
            df, 
            ind_id_col='ind_id', 
            obs_id_col='obs_id', 
            y_col='y',
            mu_x_fix_cols=None, mu_x_rnd_cols=None,
            w_x_fix_cols=None, w_x_rnd_cols=None,
            a_x_fix_cols=None, a_x_rnd_cols=None,
            nt_x_fix_cols=None, nt_x_rnd_cols=None,
            n_draws=1, 
            predict_loglik=True, t_steps=None,
            predict_rt=False
            ):
        
        ###
        #Initialise
        ###
        

        self.__initialise(
            df, 
            ind_id_col,
            obs_id_col,
            y_col,
            mu_x_fix_cols, mu_x_rnd_cols,
            w_x_fix_cols, w_x_rnd_cols,
            a_x_fix_cols, a_x_rnd_cols,
            nt_x_fix_cols, nt_x_rnd_cols,
            n_draws
            )  

        if predict_loglik and (predict_rt == False):

            ###
            #Predict log-likelihood
            ###
            
            loglik = -self.__objective_loglik(results['est'])
            print('Predictive log-lik.', loglik)
            
            return loglik
        
        elif (predict_loglik == False) and (predict_rt == False):

            ###
            #Predict probs
            ###

            probs = self.__predict_probs(results['est'], t_steps)

            return probs
        
        elif (predict_loglik == False) and predict_rt:

            ###
            #Predict response times for absorption at upper boundary
            ###

            rt = self.__predict_rt(results['est'])

            return rt

        else:

            return None
    
    def generate_data(self, N=1000, T=10):
        
        print(' ')
        print('Generating synthetic data for DDM.')
        
        ###
        #Initialise and generate data
        ###
        
        self.n_ind = N
        self.n_obs = N * T
        self.n_sit = T
        
        df = pd.DataFrame()
        df['ind_id'] = np.repeat(
            np.arange(1, self.n_ind + 1), self.n_sit
            )
        df['obs_id'] = np.arange(1, self.n_obs + 1)
        
        self.mu_x_fix_cols = ['const', 'x1']
        self.w_x_fix_cols = ['const', 'x2']
        self.a_x_fix_cols = ['const', 'x3']
        self.nt_x_fix_cols = None

        df['const'] = 1
        df['x1'] = np.random.rand(self.n_obs)
        df['x2'] = np.random.rand(self.n_obs)
        df['x3'] = np.random.rand(self.n_obs)
        
        self.mu_n_fix = len(self.mu_x_fix_cols)
        self.w_n_fix = len(self.w_x_fix_cols)
        self.a_n_fix = len(self.a_x_fix_cols)
        self.nt_n_fix = 0

        self.mu_x_fix = self.__reshape_x(df[self.mu_x_fix_cols])
        self.w_x_fix = self.__reshape_x(df[self.w_x_fix_cols])
        self.a_x_fix = self.__reshape_x(df[self.a_x_fix_cols])
        self.nt_x_fix = None

        self.mu_x_rnd_cols = ['const']
        self.w_x_rnd_cols = ['const']
        self.a_x_rnd_cols = ['const']
        self.nt_x_rnd_cols = None

        self.mu_n_rnd = 1
        self.w_n_rnd = 1
        self.a_n_rnd = 1
        self.nt_n_rnd = 0

        self.mu_x_rnd = self.__reshape_x(df[self.mu_x_rnd_cols])
        self.w_x_rnd = self.__reshape_x(df[self.w_x_rnd_cols])
        self.a_x_rnd = self.__reshape_x(df[self.a_x_rnd_cols])
        self.nt_x_rnd = 0
        
        self.n_rnd = self.mu_n_rnd + self.w_n_rnd + self.a_n_rnd + self.nt_n_rnd
        self.n_draws = 1
        self.draws = np.random.randn(self.n_ind, self.n_draws, self.n_rnd)
        
        ###
        #Set true parameter values
        ###
        
        self.__mu_beta = np.array([0.5, 0.2])
        self.__w_beta = np.array([-1.0, 0.5])
        self.__a_beta = np.array([1.5, 0.5])
        #self.__nt_beta = np.array([0])

        self.__mu_sigma = np.array([0.15])
        self.__w_sigma = np.array([0.2])
        self.__a_sigma = np.array([0.15])

        self.__mu_xi = self.__mu_sigma.reshape(1,1,self.mu_n_rnd) * self.draws[:,:,0]
        self.__w_xi = self.__w_sigma.reshape(1,1,self.w_n_rnd) * self.draws[:,:,1]
        self.__a_xi = self.__a_sigma.reshape(1,1,self.a_n_rnd) * self.draws[:,:,2] * 0

        ###
        #Generate decisions
        ###
        
        mu = self.__calc_mu()
        w = self.__calc_w()
        a = self.__calc_a()
        nt = self.__calc_nt()

        y = ddm_gen(mu, w, a, nt, self.n_ind, self.n_sit)
        y = y.reshape(-1,)
        df.insert(loc=3, column='y', value=y)
        
        return (
            df, 
            self.mu_x_fix_cols, self.mu_x_rnd_cols,
            self.w_x_fix_cols, self.w_x_rnd_cols,
            self.a_x_fix_cols, self.a_x_rnd_cols,
            self.nt_x_fix_cols, self.nt_x_rnd_cols
            )
    