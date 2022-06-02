#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy import linalg as LA
np.set_printoptions(suppress=True)
import pandas as pd
import math
import numpy.matlib
import matplotlib.pyplot as plt
import GPy
try:
    import watermark
except:
    _ = get_ipython().getoutput('pip install watermark')

try:
    import pymc3 as pm
    import theano as T
    from theano import function, shared, tensor as tt

except:
    _ = get_ipython().getoutput('conda install pymc3 theano')
    import pymc3 as pm
    import theano as T
    from theano import function, shared, tensor as tt

from pymc3.gp.util import plot_gp_dist

try:
    from pyDOE import lhs
except:
    _ = get_ipython().getoutput('pip install pyDOE')
    from pyDOE import lhs

try:
    import arviz as az
except:
    _ = get_ipython().getoutput('conda install arviz')
    import arviz as az
from google.colab import files


# generate initial training dataset
def GenerateSample(num_simulations,bounds):
    # Specify how many simulations you can afford:
    X_scaled = lhs(1, num_simulations)

    # Let's get to the original space
    X_sampled = X_scaled * (bounds[1] - bounds[0]) + bounds[0]
    X_sampled = X_sampled[:,0]
    X_sampled.sort()
    return X_sampled

def RunFunction(xx):
    return 0.5*np.exp(xx) + 0.05 * np.sin(3.*np.pi*xx)

def RunFunction1(xx):
    return 0.5*np.exp(xx) + 0.05 * np.sin(3.*np.pi*xx)

def RunFunction2(xx):
    return .5*np.exp(-xx) + 0.05 * np.cos(-3.*np.pi*xx)

class DoJobs(object):
    def __init__(self,dims,m,sigma2,testnum,num_sim,ls,s,sig,Xtr_rng,Xtr_scaled,Ytr_scaled,data_scaled,NoCh):
        self.dims = dims
        self.m = m
        self.sigma2 = sigma2
        self.testnum = testnum
        self.num_sim = num_sim
        self.ls = ls
        self.s = s
        self.sig = sig
        self.X_range = Xtr_rng
        self.Xtr_scaled = Xtr_scaled
        self.Ytr_scaled = Ytr_scaled
        self.data_scaled = data_scaled
        self.NoCh = NoCh
    def RunInference(self):

        # no. of training dataset
        X_range = np.array([-2, 2.])
        data_x = np.loadtxt('test_data_%iD_T%i.txt'%(self.dims,self.num_sim))[:,0] # true values to compare with prediction
        Xtr = np.loadtxt('x_sampled_%iD_T%i.txt'%(self.dims,self.num_sim))
        if Xtr.ndim == 1:
            Xtr_m, Xtr_s = np.mean(Xtr), np.std(Xtr)
        else:
            Xtr_m, Xtr_s = np.mean(Xtr, axis=1), np.std(Xtr,axis=1)

        Ytr = RunFunction(Xtr)
        Ytr_m, Ytr_s = np.mean(Ytr), np.std(Ytr)
        data = RunFunction(data_x)
        data_scaled = (data - Ytr_m) / Ytr_s

        # Build a surrogate model using Gaussian process regression
        # and find values of hyperparameters

        with pm.Model() as model:
            if self.Xtr_scaled.ndim == 1:
                self.Xtr_scaled = self.Xtr_scaled[:, None]

          # Define zero mean function for Gaussian process
          # zero mean function actually is default
            mean_func = pm.gp.mean.Zero()

          # Define covariance function
            cov_func = self.s**2 * pm.gp.cov.ExpQuad(input_dim=self.Xtr_scaled.shape[1], ls=self.ls) # radial basis kernel

          # Build GP
            gp = pm.gp.Marginal(mean_func, cov_func)

            y = gp.marginal_likelihood("y", X=self.Xtr_scaled, y=self.Ytr_scaled, noise = self.sig**2) # Marignal likelihood

        with model:

            # define prior of x which is solution of inverse problem
            BoundedUniform = pm.Bound(pm.Uniform, lower=-2, upper=2)
            xs = [BoundedUniform('x_%d'%(i+1), -2., 2.) for i in range(self.Xtr_scaled.shape[1])]
            x = pm.Deterministic('x', tt.stack(xs, shape=(1, self.Xtr_scaled.shape[1])))

            if self.dims == 1:
                x = x.reshape((1,1))
            if self.dims == 2:
                x = x.reshape((1,2))
            # variables passing through the inference
            # option1
            given = {"X":self.Xtr_scaled, 'y':self.Ytr_scaled, 'noise': self.sig**2, 'gp':gp}
            # option2
            givens = gp._get_given_vals(given)

            # conditional distribution; GP posterior
            mu, cov = gp._build_conditional(x,False,False,*givens)
            Mp = pm.Deterministic('Mp', mu) # mean
            if Mp.ndim == 1:
                Mp = Mp.reshape((1,1))
            Vp = pm.Deterministic('Vp', cov) # variance
            if Vp.ndim == 1:
                Vp = Vp.reshape((1,1))
            # Observation model(i.e., likelihood)
            print(self.data_scaled[self.testnum])
            y_ = pm.Normal('y_', Mp, tt.sqrt(Vp)+np.sqrt(self.sigma2), observed=self.data_scaled[self.testnum])

        with model:
            # option1: Fully Bayesian approach
            draws=2000
            trace = pm.sample(draws=draws, chains=self.NoCh, target_accept=0.99, tune=1000, progressbar=True) # target_accept=0.95, tune=1000

        summary = pm.summary(trace)
        summary_dict = summary['mean'].to_dict()
        print('By sampling\n', summary['mean'])

        temp = pm.plot_posterior(trace, var_names=['x_1'],figsize=(6,6),textsize=20);

        iii = np.argmax(temp.lines[0].get_ydata())
        x_max = temp.lines[0].get_xdata()[iii]
        print('x_max: ',x_max)

        # plot the posterior
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(temp.lines[0].get_xdata(),temp.lines[0].get_ydata(),color='b')
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel('$x_1$',fontsize=20)
        ax.set_ylabel('Probability density',fontsize=20)
        np.savetxt('PosteriorX_%iD_T%i_%ith.txt'%(self.dims,self.num_sim,self.testnum+1),np.vstack([temp.lines[0].get_xdata(),temp.lines[0].get_ydata()]).T, fmt='%.6f')
        fig.savefig('PosteriorX_%iD_T%i_%ith.pdf'%(self.dims,self.num_sim,self.testnum+1),dpi=300)

        # plot the results
        xtest = np.linspace(self.X_range[0], self.X_range[1], 200)
        xtest_scaled = (xtest - Xtr_m) / Xtr_s

        mu_all, cov_all = self.m.predict(xtest_scaled[:,None])
        dmu_dX, dcov_dX = self.m.predictive_gradients(xtest_scaled[:,None])
        dmu_dX = dmu_dX[:,:,-1]

        Sp = np.sqrt(cov_all)
        Lp = mu_all - 1.96 * Sp
        Up = mu_all + 1.96 * Sp

        mu_b, cov_b = self.m.predict(np.array([x_max])[:,None])

        Sp_b = np.sqrt(cov_b)
        Lp_b = mu_b - 1.96 * Sp_b
        Up_b = mu_b + 1.96 * Sp_b

        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.plot(xtest,mu_all*Ytr_s + Ytr_m,'b--',label='Predictive mean',zorder=3)
        ax.fill_between(xtest,Up.flatten()*Ytr_s + Ytr_m, Lp.flatten()*Ytr_s + Ytr_m, alpha=0.25, color='b', label = '95% confidential intervals',zorder=0)
        ax.plot(xtest,RunFunction(xtest),color='r',label='True function',zorder=2)
        ax.scatter(Xtr,Ytr,color='r',marker='x',label='Training dataset',zorder=5)
        ax.scatter(data_x[self.testnum],data[self.testnum],color='m',marker='o', facecolors='none',label='Test data',zorder=6)
        ax.scatter(x_max*Xtr_s + Xtr_m, mu_b*Ytr_s + Ytr_m,color='k',marker='s', facecolors='none',label='Prediction',zorder=7)
        ax.legend(frameon=True,bbox_to_anchor=(1.03,1.03),fontsize=20)
        ax.set_xlim(left=-2,right=2)
        ax.set_ylim(bottom=-.5,top=4.)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel('$X$',fontsize=20)
        ax.set_ylabel('$y$',fontsize=20)
        fig.savefig('GP_posterior_%iD_T%i_%ith.pdf'%(self.dims,self.num_sim,self.testnum+1),dpi=300)

        print('%ith inference result'%(self.testnum+1))
        print('prediction: %.6f, true: %.6f'%(x_max*Xtr_s + Xtr_m,data_x[self.testnum]))
        print('relative error: %.6f'%np.abs(((x_max*Xtr_s + Xtr_m) - data_x[self.testnum]) / data_x[self.testnum]))
        print('function value at prediction: %.6f'%(mu_b*Ytr_s + Ytr_m))
        print('true function value: %.6f'%data[self.testnum])
        np.savetxt('XYpred_%iD_T%i_%ith.txt'%(self.dims,self.num_sim,self.testnum+1), np.array([x_max*Xtr_s + Xtr_m,mu_b*Ytr_s + Ytr_m]), fmt='%.6f')

        with model:
            fcond = gp.conditional(name="fcond", Xnew=np.array([x_max])[:,None]) # pred_noise=False is default
            pred_samples = pm.sample_posterior_predictive([summary_dict], var_names=["fcond"], samples=100)
