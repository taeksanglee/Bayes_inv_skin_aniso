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

def RunFunction1(xx):
    return 0.5*np.exp(xx) + 0.05 * np.sin(3.*np.pi*xx)

def RunFunction2(xx):
    return .5*np.exp(-xx) + 0.05 * np.cos(-3.*np.pi*xx)

def RunFunction3(xx):
    return -xx**2 + 0.05 * np.cos(-3.*np.pi*xx) +4

class DoJobs(object):
    def __init__(self,dims,m1,m2,m3,sigma2,testnum,num_sim,ls,s,sig,Xtr_rng,Xtr_scaled,Ytr1_scaled,Ytr2_scaled,Ytr3_scaled,data1_scaled,data2_scaled,data3_scaled):
        self.dims = dims
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.sigma2 = sigma2
        self.testnum = testnum
        self.num_sim = num_sim
        self.ls = ls
        self.s = s
        self.sig = sig
        self.X_range = Xtr_rng
        self.Xtr_scaled = Xtr_scaled
        self.Ytr1_scaled = Ytr1_scaled
        self.Ytr2_scaled = Ytr2_scaled
        self.Ytr3_scaled = Ytr3_scaled
        self.data1_scaled = data1_scaled
        self.data2_scaled = data2_scaled
        self.data3_scaled = data3_scaled

    def RunInference(self):

        # no. of training dataset
        X_range = np.array([-2, 2.])
        data_x = np.loadtxt('test_data_%iD_T%i.txt'%(self.dims,self.num_sim))[:,0] # true values to compare with prediction
        Xtr = np.loadtxt('x_sampled_%iD_T%i.txt'%(self.dims,self.num_sim))
        if Xtr.ndim == 1:
            Xtr_m, Xtr_s = np.mean(Xtr), np.std(Xtr)
        else:
            Xtr_m, Xtr_s = np.mean(Xtr, axis=1), np.std(Xtr,axis=1)


        Ytr1 = RunFunction1(Xtr)
        Ytr1_m, Ytr1_s = np.mean(Ytr1), np.std(Ytr1)

        Ytr2 = RunFunction2(Xtr)
        Ytr2_m, Ytr2_s = np.mean(Ytr2), np.std(Ytr2)

        Ytr3 = RunFunction3(Xtr)
        Ytr3_m, Ytr3_s = np.mean(Ytr3), np.std(Ytr3)

        data1 = RunFunction1(data_x)
        data1_scaled = (data1 - Ytr1_m) / Ytr1_s

        data2 = RunFunction2(data_x)
        data2_scaled = (data2 - Ytr2_m) / Ytr2_s

        data3 = RunFunction3(data_x)
        data3_scaled = (data3 - Ytr3_m) / Ytr3_s

        # Build a surrogate model using Gaussian process regression
        # and find values of hyperparameters

        with pm.Model() as model:
            if self.Xtr_scaled.ndim == 1:
                self.Xtr_scaled = self.Xtr_scaled[:, None]

          # Define zero mean function for Gaussian process
          # zero mean function actually is default
            mean_func1 = pm.gp.mean.Zero()
            mean_func2 = pm.gp.mean.Zero()
            mean_func3 = pm.gp.mean.Zero()

          # Define covariance function
            cov_func1 = self.s[0]**2 * pm.gp.cov.ExpQuad(input_dim=self.Xtr_scaled.shape[1], ls=self.ls[0]) # radial basis kernel
            cov_func2 = self.s[1]**2 * pm.gp.cov.ExpQuad(input_dim=self.Xtr_scaled.shape[1], ls=self.ls[1])
            cov_func3 = self.s[2]**2 * pm.gp.cov.ExpQuad(input_dim=self.Xtr_scaled.shape[1], ls=self.ls[2])

          # Build GP
            gp1 = pm.gp.Marginal(mean_func1, cov_func1)
            gp2 = pm.gp.Marginal(mean_func2, cov_func2)
            gp3 = pm.gp.Marginal(mean_func3, cov_func3)

            y1 = gp1.marginal_likelihood("y1", X=self.Xtr_scaled, y=self.Ytr1_scaled, noise = self.sig[0]**2) # Marignal likelihood
            y2 = gp2.marginal_likelihood("y2", X=self.Xtr_scaled, y=self.Ytr2_scaled, noise = self.sig[1]**2)
            y3 = gp3.marginal_likelihood("y3", X=self.Xtr_scaled, y=self.Ytr3_scaled, noise = self.sig[2]**2)

        with model:

            # define prior of x which is solution of inverse problem
            BoundedUniform = pm.Bound(pm.Uniform, lower=-2., upper= 2.)
            xs = [BoundedUniform('x_%d'%(i+1), -2., 2.) for i in range(self.Xtr_scaled.shape[1])]
            x = pm.Deterministic('x', tt.stack(xs, shape=(1, self.Xtr_scaled.shape[1])))
            x = x.reshape((1,1))

            # variables passing through the inference

            given1 = {"X":self.Xtr_scaled, 'y':self.Ytr1_scaled, 'noise': self.sig[0]**2, 'gp':gp1}
            given2 = {"X":self.Xtr_scaled, 'y':self.Ytr2_scaled, 'noise': self.sig[1]**2, 'gp':gp2}
            given3 = {"X":self.Xtr_scaled, 'y':self.Ytr3_scaled, 'noise': self.sig[2]**2, 'gp':gp3}

            givens1 = gp1._get_given_vals(given1)
            givens2 = gp2._get_given_vals(given2)
            givens3 = gp3._get_given_vals(given3)

            # conditional distribution; GP posterior
            mu1, cov1 = gp1._build_conditional(x,False,False,*givens1)
            Mp1 = pm.Deterministic('Mp1', mu1[0]) # mean
            Vp1 = pm.Deterministic('Vp1', cov1[0,0]) # variance

            mu2, cov2 = gp2._build_conditional(x,False,False,*givens2)
            Mp2 = pm.Deterministic('Mp2', mu2[0]) # mean
            Vp2 = pm.Deterministic('Vp2', cov2[0,0]) # variance

            mu3, cov3 = gp3._build_conditional(x,False,False,*givens3)
            Mp3 = pm.Deterministic('Mp3', mu3[0]) # mean
            Vp3 = pm.Deterministic('Vp3', cov3[0,0]) # variance

            y_ = pm.Normal('y_', tt.stack([Mp1,Mp2,Mp3]), tt.stack([tt.sqrt(Vp1)+np.sqrt(self.sigma2),tt.sqrt(Vp2)+np.sqrt(self.sigma2),tt.sqrt(Vp3)+np.sqrt(self.sigma2)]),\
             observed=np.array([self.data1_scaled[self.testnum],self.data2_scaled[self.testnum],self.data3_scaled[self.testnum]]))

        with model:
            # option1: Fully Bayesian approach
            draws=3000
            trace = pm.sample(draws=draws, chains=6, target_accept=0.99, tune=1000, progressbar=True) # target_accept=0.95, tune=1000

        summary = pm.summary(trace)
        summary_dict = summary['mean'].to_dict()
        print('By sampling\n', summary['mean'])

        temp = pm.plot_posterior(trace, var_names=['x_1'],figsize=(6,6),textsize=20);

        iii = np.argmax(temp.lines[0].get_ydata())
        x_max = temp.lines[0].get_xdata()[iii]
        print('x_max: ',x_max)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(temp.lines[0].get_xdata(),temp.lines[0].get_ydata(),color='b')
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel('$x_1$',fontsize=20)
        ax.set_ylabel('Probability density',fontsize=20)
        np.savetxt('PosteriorX_3D_T%i_%ith.txt'%(self.num_sim,self.testnum+1),np.vstack([temp.lines[0].get_xdata(),temp.lines[0].get_ydata()]).T, fmt='%.6f')
        fig.savefig('PosteriorX_3D_T%i_%ith.pdf'%(self.num_sim,self.testnum+1),dpi=300)

        # plot the results
        xtest = np.linspace(self.X_range[0], self.X_range[1], 200)
        xtest_scaled = (xtest - Xtr_m) / Xtr_s

        mu1_all, cov1_all = self.m1.predict(xtest_scaled[:,None])
        dmu1_dX, dcov1_dX = self.m1.predictive_gradients(xtest_scaled[:,None])
        dmu1_dX = dmu1_dX[:,:,-1]

        Sp1_all = np.sqrt(cov1_all)
        Lp1_all = mu1_all - 1.96 * Sp1_all
        Up1_all = mu1_all + 1.96 * Sp1_all

        mu2_all, cov2_all = self.m2.predict(xtest_scaled[:,None])
        dmu2_dX, dcov2_dX = self.m2.predictive_gradients(xtest_scaled[:,None])
        dmu2_dX = dmu2_dX[:,:,-1]

        Sp2_all = np.sqrt(cov2_all)
        Lp2_all = mu2_all - 1.96 * Sp2_all
        Up2_all = mu2_all + 1.96 * Sp2_all

        mu3_all, cov3_all = self.m3.predict(xtest_scaled[:,None])
        dmu3_dX, dcov3_dX = self.m3.predictive_gradients(xtest_scaled[:,None])
        dmu3_dX = dmu3_dX[:,:,-1]

        Sp3_all = np.sqrt(cov3_all)
        Lp3_all = mu3_all - 1.96 * Sp3_all
        Up3_all = mu3_all + 1.96 * Sp3_all

        mu1_b, cov1_b = self.m1.predict(np.array([x_max])[:,None])
        mu2_b, cov2_b = self.m2.predict(np.array([x_max])[:,None])
        mu3_b, cov3_b = self.m3.predict(np.array([x_max])[:,None])

        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.plot(xtest,RunFunction1(xtest),color='r',label='True function #1',zorder=2)
        ax.plot(xtest,RunFunction2(xtest),color='m',label='True function #2',zorder=2)
        ax.plot(xtest,RunFunction3(xtest),color='gray',label='True function #3',zorder=2)
        ax.plot(xtest,mu1_all*Ytr1_s + Ytr1_m,'b--',label='Predictive mean #1',zorder=3)
        ax.fill_between(xtest,Up1_all.flatten()*Ytr1_s + Ytr1_m, Lp1_all.flatten()*Ytr1_s + Ytr1_m, alpha=0.25, color='b', label = '95% confidential intervals #1',zorder=0)
        ax.plot(xtest,mu2_all*Ytr2_s + Ytr2_m,'g--',label='Predictive mean #2',zorder=3)
        ax.fill_between(xtest,Up2_all.flatten()*Ytr2_s + Ytr2_m, Lp2_all.flatten()*Ytr2_s + Ytr2_m, alpha=0.25, color='g', label = '95% confidential intervals #2',zorder=0)
        ax.plot(xtest,mu3_all*Ytr3_s + Ytr3_m,'y--',label='Predictive mean #3',zorder=3)
        ax.fill_between(xtest,Up3_all.flatten()*Ytr3_s + Ytr3_m, Lp3_all.flatten()*Ytr3_s + Ytr3_m, alpha=0.25, color='y', label = '95% confidential intervals #3',zorder=0)
        # plot the data and the true latent function

        ax.scatter(Xtr,Ytr1,color='r',marker='x',label='Training dataset #1',zorder=5)
        ax.scatter(Xtr,Ytr2,color='m',marker='x',label='Training dataset #2',zorder=5)
        ax.scatter(Xtr,Ytr3,color='gray',marker='x',label='Training dataset #3',zorder=5)

        ax.scatter(data_x[self.testnum],data1[self.testnum],color='brown',marker='o', facecolors='none',label='Test data #1',zorder=6)
        ax.scatter(data_x[self.testnum],data2[self.testnum],color='brown',marker='^', facecolors='none',label='Test data #2',zorder=6)
        ax.scatter(data_x[self.testnum],data3[self.testnum],color='brown',marker='v', facecolors='none',label='Test data #3',zorder=6)
        ax.scatter(x_max*Xtr_s + Xtr_m, mu1_b*Ytr1_s + Ytr1_m,color='k',marker='s', facecolors='none',label='Prediction #1',zorder=7)
        ax.scatter(x_max*Xtr_s + Xtr_m, mu2_b*Ytr2_s + Ytr2_m,color='k',marker='d', facecolors='none',label='Prediction #2',zorder=7)
        ax.scatter(x_max*Xtr_s + Xtr_m, mu3_b*Ytr3_s + Ytr3_m,color='k',marker='h', facecolors='none',label='Prediction #3',zorder=7)
        ax.legend(frameon=True,bbox_to_anchor=(1.03,1.03),fontsize=20)
        ax.set_xlim(left=-2,right=2)
        ax.set_ylim(bottom=-0.5,top=4.5)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel('$X$',fontsize=20)
        ax.set_ylabel('$y$',fontsize=20)
        fig.savefig('GP_posterior_3D_T%i_%ith.pdf'%(self.num_sim,self.testnum+1),dpi=300)

        print('%ith inference result'%(self.testnum+1))
        print('prediction: %.6f, true: %.6f'%(x_max*Xtr_s + Xtr_m,data_x[self.testnum]))
        print('relative error: %.6f'%np.abs(((x_max*Xtr_s + Xtr_m) - data_x[self.testnum]) / data_x[self.testnum]))
        print('function1 value at prediction: %.6f'%(mu1_b*Ytr1_s + Ytr1_m))
        print('function2 value at prediction: %.6f'%(mu2_b*Ytr2_s + Ytr2_m))
        print('function3 value at prediction: %.6f'%(mu3_b*Ytr3_s + Ytr3_m))
        print('true1 function value: %.6f'%data1[self.testnum])
        print('true2 function value: %.6f'%data2[self.testnum])
        print('true3 function value: %.6f'%data3[self.testnum])

        np.savetxt('XYpred_%iD_T%i_%ith.txt'%(self.dims,self.num_sim,self.testnum+1), np.array([x_max*Xtr_s + Xtr_m,mu1_b*Ytr1_s + Ytr1_m,mu2_b*Ytr2_s + Ytr2_m, mu3_b*Ytr3_s + Ytr3_m]), fmt='%.6f')
