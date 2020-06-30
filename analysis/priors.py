# -*- coding: utf-8 -*-

from random import random
from math import floor

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
from scipy.special import loggamma
from scipy.stats import gaussian_kde, norm, truncnorm
from openturns import Gamma, MaximumEntropyOrderStatisticsDistribution

import util

#=============================================================================#


class PriorTheta:
    def __init__(
            self,
            name,
            colour,
            target,
            transformation=lambda X: X,
            full_cond = {"prior": None, "post": None},
            para_names = util.para_names["theta"],
            inst_name=""):
        """
        Prior for (mu, sigma, xi), sampled using MCMC
        name:           name given in report
        inst_name:      name for a specific instance of a prior
        colour:         colour used in report
        target:         target distribution for MCMC
        transformation: transformation of target distribution which outputs
                        prior for theta
        full_cond:      full conditionals for MCMC algorithm
        """
        self.name = name
        self.inst_name = inst_name
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
            save=False):
        """
        MCMC algorithm for theta
        Calculates marginals using KDE with the sample
        Sets analytic marginals if provided
        Marginals for theta are use log(sigma) parametrisation,
        and must be specified in log form
        typ:     "prior" or "post"
        init:    initialisation for MCM algorithm
        prop_sd: proposal standard deviations for MCMC algorithm
        data:    GEVdata for posterior

        """

        if getattr(self, typ)["proper"]:
            # If proper
            
            # Number of iterations
            N = 11000
            
            # Length of burn-in
            burnin = 1000
            
            # Log density
            def kde(sample):
                k = gaussian_kde(np.transpose(sample))
                def f(X):
                    return k.logpdf(X)[0]
                return f
            
            if typ == "post":
                self.post["data"] = data
                
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
                col=util.pal[self.colour],
                save=save,
                save_name="%s-%s-%s" % (
                    self.inst_name,
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
        else:
            # If improper
            
            getattr(self, typ)["theta"]["marginal"] = [
                None
                for I in util.marg_1_2]
            getattr(self, typ)["q"]["marginal"] = [
                None
                for I in util.marg_1_2]
  
        try:
            # Set analytic marginals
            self.set_marginals()
        except:
            pass
            
            
class PriorQ(PriorTheta):
    def __init__(self, name, colour, p, q, inst_name=""):
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
                "post": lambda X, data: util.logpost(X, prior_theta, data)},
            inst_name=inst_name)
    
        
class PriorG3(PriorQ):
    def __init__(self, p, para, inst_name=""):
        """
        Gamma prior distributions for three quantile differences
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
    
        super().__init__("G3", 0, p, q, inst_name=inst_name)
        
    def set_marginals(self):
        self.prior["theta"]["marginal"][2] = self.marg_xi
        self.prior["theta"]["marginal"][5] = self.marg_logsigma_xi
        
            
class PriorMEC(PriorQ):
    def __init__(self, p, para, quantile_diff=False, inst_name="", save=False):
        """
        Gamma prior distributions for three
            quantiles with maximum entropy copula
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
            
            for i in [2, 1, 0]:
                if i == 2:
                    def bound_para(x, y):
                        x_new = float(abs(x))
                        y_new = float(abs(y))
                        return (x_new, y_new)
                    init = (100.0, 1)
                else:
                    x_old = q_marg_gamma[i + 1].getParameter()[0]
                    y_old = q_marg_gamma[i + 1].getParameter()[1]
                    def bound_para(x, y):
                        x_new = x_old - float(abs(x))
                        y_new = y_old + float(abs(y))
                        if x_new <= 0:
                            x_new = 1e-10
                        return (x_new, y_new)
                    init = (x_old / 2.0, y_old / 2.0)
                    
                def KL(para):
                    para = bound_para(*para)
                    
                    dist = Gamma(*para, 0.0)
                    
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
                
                res = minimize(KL, init, tol=10)
                
                k, ld = bound_para(*res.x)
                
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
                
                fig, ax = plt.subplots()
                X = np.linspace(0, 400, 400)
                ax.plot(
                    X,
                    [q_marg[i].computePDF(x) for x in X],
                    label="G3 prior")
                ax.plot(
                    X,
                    [q_marg_gamma[i].computePDF(x) for x in X],
                    label="MEC prior")
                
                ax.set(xlabel=util.para_names["q"][i])
                ax.grid()
                        
                ax.legend(
                    ncol=2,
                    bbox_to_anchor=(0.5, 1),
                    loc="lower center",
                    frameon=False)
                
                fig.set_size_inches(2.5, 2.5)
                if save:
                    plt.savefig(
                        "%s/plots/%s-MEC-approx-%s.pdf" % (
                            util.save_path,
                            inst_name,
                            i),
                        bbox_inches="tight")
                plt.show()
                
            print(details)
        else:
            q_marg_gamma = [Gamma(*para[i], 0.0) for i in range(3)]
            
        self.q_marg_gamma = q_marg_gamma
                
        ot = MaximumEntropyOrderStatisticsDistribution(q_marg_gamma)
        
        def q(X):
            if np.any(np.array([X[0], X[1] - X[0], X[2] - X[1]]) <= 0.0):
                return None
            
            Y = ot.computePDF(X)
            
            # Sometimes the density is negative
            if Y <= 0:
                if Y < 0:
                    print("Negative density at", X)
                return None
            
            return np.log(Y)
        
        super().__init__("MEC", 1, p, q, inst_name=inst_name)


class PriorG2(PriorTheta):
    def __init__(self, p, para, inst_name=""):
        """
        Gamma prior distributions for two quantile differences
        p:             probabilities (p_1, p_2)
        para:          list of Gamma parameters (shape and rate) of prior for
                       quantile differences (~q_1, ~q_2)
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
        
        # Transformation (mu, theta, xi) -> (sigma, q1, q2)
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
            "G2",
            2,
            {
                "prior": prior_theta,
                "post": lambda X, data: util.logpost(X, prior_theta, data)},
            inst_name=inst_name)
        
        self.prior["proper"] = False
        
        def marg_xi(X):
            xi = X[0]
            
            e = np.exp(s * xi)
            
            y = abs(((s[1] * e[1] - s[0] * e[0]) / (e[1] - e[0])) - (1 / xi))
            
            return np.log(y) - 2.0
        
        self.marg_xi = marg_xi
        
    def set_marginals(self):
        # Improper uniform prior
        self.prior["theta"]["marginal"][1] = lambda X: np.log(0.1)
        
        self.prior["theta"]["marginal"][2] = self.marg_xi
        
        
class PriorG1(PriorTheta):
    def __init__(self, p, para, inst_name=""):
        """
        Gamma prior distributions for one quantile difference
        p:             probability p_1
        para:          Gamma parameters (shape and rate) of prior for
                       quantile q_1
                       [alpha_1, beta_1]
        """
        
        self.p = np.array(p)
        
        s = -np.log(-np.log(1.0 - self.p[0]))
        
        # This constant does not need to be exact
        C = para[0] * np.log(para[1]) - loggamma(para[0])
        
        # Prior of (sigma, mu, q1)
        def f(X):
            sigma, mu, q1 = X
            
            if sigma <= 0:
                return None
            
            if q1 <= 0.0:
                return None
            
            return (
                C
                + (para[0] - 1.0) * np.log(q1) - para[1] * q1
                - np.log(sigma))
        
        # Transformation (mu, theta, xi) -> (mu, sigma, q1)
        def g(X):
            mu, sigma, xi = X
            
            # When xi is close enough to 0, we consider it equal to 0
            if abs(xi) < 1e-300:
                q1 = mu + sigma * s
            else:
                q1 = mu + sigma * (np.exp(xi * s) - 1.0) / xi
        
            if q1 < 0.0:
                return None
            
            return [mu, sigma, q1]
    
    
        # Log of determinant of g
        def g_det(X):
            mu, sigma, xi = X
            
            if abs(xi) < 1e-300:
                return np.log(sigma)
                
            e = (s * xi - 1.0) * np.exp(s * xi) + 1
        
            return np.log(sigma) + np.log(e) - np.log(xi ** 2.0)
        
        prior_theta = util.log_transform(f, g, g_det)
            
        super().__init__(
            "G1",
            2,
            {
                "prior": prior_theta,
                "post": lambda X, data: util.logpost(X, prior_theta, data)},
            inst_name=inst_name)
        
        self.prior["proper"] = False
        
    def set_marginals(self):
        # Improper uniform priors
        self.prior["theta"]["marginal"][0] = lambda X: np.log(0.01)
        self.prior["theta"]["marginal"][1] = lambda X: np.log(0.1)
        

class PriorSAS(PriorTheta):
    def __init__(self, prior, alpha):
        """
        Spike-and-slab prior distribution for xi
        prior: PriorQD
        """
        self.p = prior.p
        self.alpha = alpha
        
        
        # (mu, sigma, beta, gamma) -> (mu, sigma, xi)
        def trans(X):
            return [X[0], X[1], X[2] * X[3]]
        
        
        def prior_eta(X):
            bern = alpha ** X[3] * (1.0 - alpha) ** (1.0 - X[3])
            return np.log(bern) + prior.target["prior"]([X[0], X[1], X[2]])
        
        
        def post_eta(X, data):
            return util.loglik(trans(X), data) + prior_eta(X)
        
        
        def prior_gamma_full_cond(X):
            return int(random() < alpha)
            
        
        def post_gamma_full_cond(X, data):
            mu, sigma, beta = X
            
            l1 = util.loglik(X, data)
            l2 = util.loglik([mu, sigma, 0], data)
            
            # Prevents rounding errors
            if abs(l2 - l1) > 500.0:
                return(int(l1 < l2))
            
            m = min(l1, l2)
            
            C = -m - 500.0
            
            e1 = alpha * np.exp(l1 + C)
            e2 = (1.0 - alpha) * np.exp(l2 + C)
            
            p = e1 / (e1 + e2)
            
            return int(random() < p)
            
        super().__init__(
            "SAS",
            3,
            {"prior": prior_eta, "post": post_eta},
            transformation=trans,
            full_cond={
                "prior": [None, None, None, prior_gamma_full_cond],
                "post": [None, None, None, post_gamma_full_cond]},
            para_names=[r"$\mu$", r"$\log(\sigma)$", r"$\beta$", r"$\gamma$"],
            inst_name=prior.inst_name)
     

class PriorE(PriorG3):
    def __init__(self, p, means, inst_name=""):
        """
        Exponential prior distributions for quantile differences
        p:      probabilities (p_1, p_2, p_3)  
        means: list of means of
               quantile differences (~q_1, ~q_2, ~q_3)
        """
    
        super().__init__(
            p,
            [[1.0, 1.0 / m] for m in means],
            inst_name=inst_name)
        
        self.name = "E"
        self.colour = 4
    

class PriorTN(PriorQ):
    def __init__(self, p, para, inst_name=""):
        """
        Truncated normal prior distributions for quantile differences
        p:      probabilities (p_1, p_2, p_3)  
        para: list of [mean, variance] of
              quantile differences (~q_1, ~q_2, ~q_3)
        """
        
        def tn_para(m, V):
            """
            Converts mean and variance to truncated normal parameters
            m: mean
            V: variance
            Outputs: (parent mean, parent standard deviation)
            """
            def hazard(x):
                return norm.pdf(x) / norm.sf(x)
                
            def f(x):
                h = hazard(x)
                return (1 + x * h - h ** 2) / (h - x) ** 2
        
            def g(x):
                return f(x) - (V / m ** 2)
            
            def g_prime(x):
                h = hazard(x)
                d = h - x
                return (2 + h * d * (-3 + d * x)) / d ** 3
            
            path = "%s/latex-bits/%s-TN-target.txt" % (
                util.save_path,
                inst_name)
            with open(path, "w") as text_file:
                print(
                    "$(%s, %s)$" % (util.latex_f(m), util.latex_f(V)),
                    file=text_file)
            
            sol = root_scalar(g, x0=0, bracket=(-20, 20))
            
            alpha = sol.root
        
            sigma = m / (hazard(alpha) - alpha)
            mu = -alpha * sigma
            
            stats = truncnorm.stats(alpha, np.inf, mu, sigma, moments="mv")
            
            path = "%s/latex-bits/%s-TN-approx.txt" % (
                util.save_path,
                inst_name)
            with open(path, "w") as text_file:
                print(
                    "$(%s)$" % ",".join([util.latex_f(x) for x in stats]),
                    file=text_file)
            
            return (mu, sigma)
        
        para = np.array([tn_para(x, y) for x, y in para])
        
        # X = (q1, q2, q3)
        def q(X):
            Y = np.array([X[0], X[1] - X[0], X[2] - X[1]])
            if np.any(Y <= 0.0):
                return None
            
            return np.sum(np.log(truncnorm.pdf(
                Y,
                -para[:, 0] / para[:, 1],
                np.inf,
                loc=para[:, 0],
                scale=para[:, 1])))
        
        super().__init__("TN", 5, p, q, inst_name=inst_name)
    
        
def all_priors(
        p,
        alpha,
        para=None,
        theta=None,
        variance=None,
        name="",
        save=False):
    """
    Returns a list of all priors
    Requires either Gamma parameters for prior for quantile differences,
    or a theta=[mu, sigma, xi] and variance from which Gamma parameters
    are derived
    p:            (p_1, p_2, p_3)
    para:         Gamma parameters for prior for quantile differences
    theta:        List [mu, sigma, xi] which Gamma distributions are
                  centered on
    variance:     variance of Gamma distributions
    name:         gives instance names for the priors
    """
    
    p = np.array(p)
    
    if theta is not None:
        qu = util.quantile(theta, 1.0 - p)
        
        qu_diff = np.array([qu[0], qu[1] - qu[0], qu[2] - qu[1]])
        print("q~:", [util.latex_f(x) for x in qu_diff])
        
        para = np.vstack(
            (qu_diff ** 2 / variance, qu_diff / variance)).T
        
    qd = PriorG3(p, para, inst_name=name)
    
    return [
        qd,
        PriorG2(
            p,
            para[:2],
            inst_name=name),
        PriorG1(
            p,
            para[:1][0],
            inst_name=name),
        PriorMEC(
            p,
            para,
            quantile_diff=True,
            inst_name=name,
            save=save),
        PriorE(
            p,
            means=[x / y for x, y in para],
            inst_name=name),
        PriorTN(
            p,
            para=[[x / y, x / y ** 2] for x, y in para],
            inst_name=name)]