# -*- coding: utf-8 -*-

from random import random

import numpy as np
from scipy.optimize import minimize
from scipy.special import loggamma
from scipy.stats import gaussian_kde, truncnorm, expon, gamma
from openturns import Gamma, MaximumEntropyOrderStatisticsDistribution

import util


class PriorTheta:
    def __init__(
            self,
            name,
            colour,
            target,
            transformation=lambda X: X,
            full_cond = {"prior": None, "post": None},
            para_names = util.para_names["theta"]):
        """
        Prior for (mu, sigma, xi), sampled using MCMC
        name:           name given in report
        colour:         colour used in report
        target:         target distribution for MCMC
        transformation: transformation of target distribution which outputs prior
                        for theta
        full_cond:      full conditionals for MCMC algorithm
        """
        self.name = name
        self.colour = colour
        
        # Stores whether the univariate marginals are proper distributions        
        self.prior = {
            "q": {},
            "theta": {},
            "proper": True}
        
        self.post = {
            "q": {},
            "theta": {},
            "proper": True}
            
        self.target = target
        self.transformation = transformation
        self.full_cond = full_cond
        self.para_names = para_names
        
            
    def mcmc(
            self,
            typ,
            init,
            prop_sd,
            data=None,
            save=False,
            save_name=""):
        """
        MCMC algorithm for theta
        typ:     "prior" or "post"
        init:    initialisation for MCM algorithm
        prop_sd: proposal standard deviations for MCMC algorithm
        data:    GEVdata for posterior

        """
        if not getattr(self, typ)["proper"]:
            return
        
        # Number of iterations
        N = 101000
        
        # Length of burn-in
        burnin = 1000
        
        # Log density
        def kde(sample):
            k = gaussian_kde(np.transpose(sample))
            def f(X):
                return k.logpdf(X)[0]
            return f
        
        if typ == "post":
            target = lambda X: self.target["post"](X, data)
            if self.full_cond[typ] is None:
                fc = None
            else:
                fc = self.full_cond[typ].copy()
                for i, f in enumerate(self.full_cond["post"]):
                    if f is not None:
                        fc[i] = lambda X: f(X, data)
                
        else:
            target = self.target["prior"]
            fc = self.full_cond["prior"]
        
        mcmc_obj = util.MCMCSample(
            util.log_transform(target, util.sig_trans, lambda X: X[1]),
            init,
            prop_sd,
            N,
            burnin,
            full_cond=fc)
        
        mcmc_obj.results(
            para_names=self.para_names,
            save=save,
            save_name="%s-%s-%s" % (
                save_name,
                self.name,
                typ))
        
        sample_theta = np.array([
            self.transformation(X)
            for X in mcmc_obj.sample])
        
        getattr(self, typ)["theta"]["sample"] = sample_theta
        getattr(self, typ)["theta"]["marginal"] = [
            kde(sample_theta[:, I])
            for I in util.marg_1_2]
        
        # Transformation log(sigma) -> sigma:
        
        sample_theta[:, 1] = np.exp(sample_theta[:, 1])
        
        sample_q = np.array([
            util.quantile(theta, 1.0 - self.p)
            for theta in sample_theta])
        
        getattr(self, typ)["q"]["sample"] = sample_q
        getattr(self, typ)["q"]["marginal"] = [
            kde(sample_q[:, I])
            for I in util.marg_1_2]
        
        getattr(self, typ)["mcmc"] = mcmc_obj
        
            
class PriorQ(PriorTheta):
    def __init__(self, name, colour, p, q):
        """
        Log prior for quantiles (q_1, q_2, q_3)
        name:   name given in report
        colour: colour used in report
        p:      probabilities (p_1, p_2, p_3)
        q:      joint PDF of (q_1, q_2, q_3)
        """
        self.p = np.array(p)
        self.s = -np.log(-np.log(1.0 - self.p))
        
        # Transformation (mu, theta, xi) -> (q1, q2, q3)
        def g(X):
            mu, sigma, xi = X
            
            if sigma <= 0:
                return None
            
            # When xi is close enough to 0, we consider it equal to 0
            if abs(xi) < 1e-300:
                q = mu + sigma * self.s
            else:
                q = mu + sigma * (np.exp(xi * self.s) - 1.0) / xi
        
            if q[0] < 0.0:
                return None
            return q
    
    
        # Log of determinant of g
        def g_det(X):
            mu, sigma, xi = X
            
            if abs(xi) < 1e-300:
                return np.log(sigma)
                
            e = np.exp(self.s * xi)
            
            sm = [
                self.s[i] * e[i] * (e[(i + 2) % 3] - e[(i + 1) % 3])
                for i in range(3)]
                
            return np.log(sigma) + np.log(sum(sm)) - np.log(xi ** 2.0)
        
        prior_theta = util.log_transform(q, g, g_det)
        
        super().__init__(
            name,
            colour,
            {
                "prior": prior_theta,
                "post": lambda X, data: util.logpost(X, prior_theta, data)})
    
        
class PriorQD(PriorQ):
    def __init__(self, p, para):
        """
        Prior from independent quantile differences
        p:      probabilities (p_1, p_2, p_3)  
        para: list of Gamma parameters (shape and rate) of prior for
              quantile differences (~q_1, ~q_2, ~q_3)
              [[alpha_1, beta_1], [alpha_2, beta_2], [alpha_3, beta_3]]
        """
        
        para = np.array(para)
        
        # This constant does not need to be exact
        C = np.sum(para[:, 0] * np.log(para[:, 1]) - loggamma(para[:, 0]))
        
        # X = (q1, q2, q3)
        def q(X):
            Y = np.array([X[0], X[1] - X[0], X[2] - X[1]])
            if np.any(Y <= 0.0):
                return None
            
            return C + np.sum((para[:, 0] - 1.0) * np.log(Y) - para[:, 1] * Y)
        
        def log_Phi1(xi):
            e = np.exp(self.s * xi)
    
            sm = sum([
                self.s[i] * e[i] * (e[(i + 2) % 3] - e[(i + 1) % 3])
                for i in range(3)])
            
            return (
                (para[1, 0] - 1.0) * np.log((e[1] - e[0]) / xi)
                + (para[2, 0] - 1.0) * np.log((e[2] - e[1]) / xi) 
                + np.log(sm)
                - np.log(xi ** 2))
        
        def Phi2(xi):
            e = np.exp(self.s * xi)
            
            sm = para[1, 1] * (e[1] - e[0]) + para[2, 1] * (e[2] - e[1])
            
            return sm / xi
        
        
        def marg_xi(X):
            xi = X[0]
            
            if abs(xi) < 1e-5:
                xi = 1e-5
            
            C = (loggamma(para[1, 0] + para[2, 0])
                + para[1, 0] * np.log(para[1, 1])
                + para[2, 0] * np.log(para[2, 1])
                - loggamma(para[1, 0])
                - loggamma(para[2, 0]))
            
            res = (
                C 
                + log_Phi1(xi)
                - (para[1, 0] + para[2, 0]) * np.log(Phi2(xi)))
                
            return res
        
    
        def marg_sigma_xi(X):
            sigma, xi = X
            
            if sigma <= 0:
                return -np.inf
            
            if abs(xi) < 1e-5:
                xi = 1e-5
            
            C = (para[1, 0] * np.log(para[1, 1])
                 + para[2, 0] * np.log(para[2, 1])
                 - loggamma(para[1, 0])
                 - loggamma(para[2, 0]))
            
            res = (
                C
                + (para[1, 0] + para[2, 0] - 1) * np.log(sigma) 
                + log_Phi1(xi)
                - sigma * Phi2(xi))
                
            return res
        
        self.marg_xi = marg_xi
        self.marg_logsigma_xi = util.log_transform(
            marg_sigma_xi,
            util.sig_trans,
            lambda X: X[1])
    
        super().__init__("QD", 0, p, q)
        
    def set_marginals(self):
        self.prior["theta"]["marginal"][2] = self.marg_xi
        self.prior["theta"]["marginal"][5] = self.marg_logsigma_xi
            
class PriorME(PriorQ):
    def __init__(self, p, para, quantile_diff=False):
        """
        Prior from maximum entropy distribution of quantiles
        p:             probabilities (p_1, p_2, p_3)  
        para:          list of Gamma parameters (shape and rate) of prior for
                       quantiles (q_1, q_2, q_3)
                       [
                           [alpha_1, beta_1],
                           [alpha_2, beta_2],
                           [alpha_3, beta_3]]
        quantile_diff: if True, para are parameters for quantile differences,
                       and parameters for quantiles are estimated
        """
        
        para = np.array(para)
        
        if quantile_diff:
            details = ""
            
            # Marginals of q_tilde
            q_tilde_marg = [Gamma(*para[i], 0.0) for i in range(3)]
            
            # Marginals of q
            q_marg = [
                q_tilde_marg[0],
                q_tilde_marg[0] + q_tilde_marg[1],
                q_tilde_marg[0] + q_tilde_marg[1] + q_tilde_marg[2]]
            
            q_marg_gamma = [None for _ in range(3)]
            
            for i in range(3):
                def KL(para):
                    _k, _ld = para
                    
                    _k = float(abs(_k))
                    _ld = float(abs(_ld))
                    
                    dist = Gamma(_k, _ld, 0.0)
                    
                    upper_bound = max(
                        q_marg[i].getRange().getUpperBound()[0],
                        dist.getRange().getUpperBound()[0])
                    
                    def integrand(x):
                        pdf1 = dist.computePDF(x)
                        pdf2 = q_marg[i].computePDF(x)
                        pdf1 = max(pdf1, 1e-140)
                        pdf2 = max(pdf2, 1e-140)
                        return pdf1 * np.log(pdf1 / pdf2)
                    
                    X = np.linspace(0.0, upper_bound, 50)
                    return upper_bound * np.mean([integrand(x) for x in X])
                
                res = minimize(KL, (100.0, 0.1))
                
                k, ld = res.x
                
                k = float(abs(k))
                ld = float(abs(ld))
                
                q_marg_gamma[i] = Gamma(k, ld, 0.0)
                
                s = [
                    "alpha = %s, beta = %s" % (k, ld),
                    "q_exact %d: median = %.3f, 0.95-quantile = %.3f" % (
                        i,
                        q_marg[i].computeQuantile(0.5)[0],
                        q_marg[i].computeQuantile(0.95)[0]),
                    "q_gamma %d: median = %.3f, 0.95-quantile = %.3f" % (
                        i,
                        q_marg_gamma[i].computeQuantile(0.5)[0],
                        q_marg_gamma[i].computeQuantile(0.95)[0])]
                
                details += "\n".join(s) + "\n"
                
            print(details)
                
            self.details = lambda: print(details)
        else:
            q_marg_gamma = [Gamma(*para[i], 0.0) for i in range(3)]
                
        ot = MaximumEntropyOrderStatisticsDistribution(q_marg_gamma)
        
        def q(X):
            if np.any(np.array([X[0], X[1] - X[0], X[2] - X[1]]) <= 0.0):
                return None
            
            Y = ot.computePDF(X)
            
            # Sometimes the density is negative
            if Y <= 0:
                return None
            
            return np.log(Y)
        
        super().__init__("ME", 1, p, q)


class PriorULS(PriorTheta):
    def __init__(self, p, para):
        """
        Uniform prior on log sigma
        p:             probabilities (p_1, p_2)  
        para:          list of Gamma parameters (shape and rate) of prior for
                       quantiles (q_1, q_2)
                       [
                           [alpha_1, beta_1],
                           [alpha_2, beta_2]]
        """
        
        self.p = np.array(p)
        para = np.array(para)
        
        s = -np.log(-np.log(1.0 - self.p[:2]))
        
        # This constant does not need to be exact
        C = np.sum(para[:, 0] * np.log(para[:, 1]) - loggamma(para[:, 0]))
        
        # Prior of (sigma, q1, q2)
        def f(X):
            sigma, q1, q2 = X
            
            if sigma <= 0:
                return None
            
            Y = np.array([q1, q2 - q1])
            if np.any(Y <= 0.0):
                return None
            
            return (
                C
                + np.sum((para[:, 0] - 1.0) * np.log(Y) - para[:, 1] * Y)
                - np.log(sigma))
            
        def g(X):
            mu, sigma, xi = X
            
            # When xi is close enough to 0, we consider it equal to 0
            if abs(xi) < 1e-300:
                q = mu + sigma * s
            else:
                q = mu + sigma * (np.exp(xi * s) - 1.0) / xi
        
            if q[0] < 0.0:
                return None
            
            return np.concatenate(([sigma], q))
    
    
        # Log of determinant of g
        def g_det(X):
            mu, sigma, xi = X
            
            if abs(xi) < 1e-300:
                return np.log(sigma)
                
            e = (s * xi - 1.0) * np.exp(s * xi)
        
            return np.log(sigma) + np.log(abs(e[0] - e[1])) - np.log(xi ** 2.0)
        
        prior_theta = util.log_transform(f, g, g_det)
            
        super().__init__(
            "ULS",
            2,
            {
                "prior": prior_theta,
                "post": lambda X, data: util.logpost(X, prior_theta, data)})
        
        self.prior["proper"] = False
        

class PriorSAS(PriorTheta):
    def __init__(self, prior):
        """
        Spike-and-slab prior
        prior: PriorQD
        """
        self.p = prior.p
        
        
        # (mu, sigma, beta, gamma) -> (mu, sigma, xi)
        def trans(X):
            return [X[0], X[1], X[2] * X[3]]
        
        
        def prior_eta(X):
            return np.log(0.5) + prior.target["prior"]([X[0], X[1], X[2]])
        
        
        def post_eta(X, data):
            return util.loglik(trans(X), data) + prior_eta(X)
        
        
        def prior_gamma_full_cond(X):
            return int(random() < 0.5)
            
        
        def post_gamma_full_cond(X, data):
            mu, sigma, beta = X
            
            l1 = util.loglik(X, data)
            l2 = util.loglik([mu, sigma, 0], data)
            
            # Prevents rounding errors
            if abs(l2 - l1) > 500.0:
                return(int(l1 < l2))
            
            m = min(l1, l2)
            
            C = -m - 500.0
            
            p = np.exp(l1 + C) / (np.exp(l1 + C) + np.exp(l2 + C))
            
            return int(random() < p)
            
        super().__init__(
            "SAS",
            3,
            {"prior": prior_eta, "post": post_eta},
            transformation=trans,
            full_cond={
                "prior": [None, None, None, prior_gamma_full_cond],
                "post": [None, None, None, post_gamma_full_cond]},
            para_names=[r"$\mu$", r"$\sigma$", r"$\beta$", r"$\gamma$"])
     

class PriorQDE(PriorQD):
    def __init__(self, p, means):
        """
        Prior 5
        p:      probabilities (p_1, p_2, p_3)  
        means: list of means of prior for
               quantile differences (~q_1, ~q_2, ~q_3)
        """
    
        super().__init__(p, [[1.0, 1.0 / m] for m in means])
        
        self.name = "QDE"
        self.colour = 4
        

class PriorQDTN(PriorQ):
    def __init__(self, p, para):
        """
        Prior 6
        p:      probabilities (p_1, p_2, p_3)  
        para: list of Gamma parameters (shape and rate) of prior for
              quantile differences (~q_1, ~q_2, ~q_3)
              [[alpha_1, beta_1], [alpha_2, beta_2], [alpha_3, beta_3]]
        """
        
        para = np.array(para)
        
        if False:
            par = [None for _ in range(3)]
            
            for i in range(3):
                target = lambda x: gamma.pdf(x, para[i][0], scale=1.0 / para[i][1])
                
                def KL(par):
                    mu, sigma = par
                    
                    mu = float(abs(mu))
                    sigma = float(abs(sigma))
                    
                    dist = lambda x: truncnorm.pdf(x, 0, np.inf, loc=mu, scale=sigma)
                    
                    upper_bound = 100
                    
                    def integrand(x):
                        pdf1 = dist(x)
                        pdf2 = target(x)
                        pdf1 = max(pdf1, 1e-140)
                        pdf2 = max(pdf2, 1e-140)
                        return pdf1 * np.log(pdf1 / pdf2)
                    
                    X = np.linspace(0.0, upper_bound, 50)
                    return upper_bound * np.mean([integrand(x) for x in X])
                
                res = minimize(KL, (0.0, 1.0))
                
                mu, sigma = res.x
                
                mu = float(abs(mu))
                sigma = float(abs(sigma))
                
                print([mu, sigma])
                
                par[i] = [mu, sigma]
        else:
            par = para
        
        # X = (q1, q2, q3)
        def q(X):
            Y = np.array([X[0], X[1] - X[0], X[2] - X[1]])
            if np.any(Y <= 0.0):
                return None
            
            return np.sum(np.log(truncnorm.pdf(Y, -par[:, 0] / par[:, 1], np.inf, loc=par[:, 0], scale=par[:, 1])))
        
        super().__init__("P6", 5, p, q)
        
def all_priors(p, para_qu_diff):
    """
    Returns a list of all priors
    p:            (p_1, p_2, p_3)
    para_qu_diff: Gamma parameters for prior for quantile differences
    """
    qd = PriorQD(p, para_qu_diff)
    
    return [
        qd,
        PriorME(p, para_qu_diff, quantile_diff=True),
        PriorULS(p, para_qu_diff[:2]),
        PriorSAS(qd),
        PriorQDE(p, means=[x / y for x, y in para_qu_diff]),
        PriorQDTN(p, para=[[x / y, x / y ** 2] for x, y in para_qu_diff])]