# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import gaussian_kde
from openturns import Normal, MaximumEntropyOrderStatisticsDistribution, TruncatedDistribution

import util

#=============================================================================#


class PriorTheta:
    def __init__(
            self,
            pdf,
            colour,
            inst_name=""):
        """
        Prior for (mu, sigma, xi).
        -----------------------------------------------------------------------
        pdf:       Prior PDF (function).
        colour:    Index of some colour in util.pal.
        inst_name: Label.
        -----------------------------------------------------------------------
        Returns: PriorTheta object.
        """
        
        self.colour = colour
        self.inst_name = inst_name
        
        # Stores whether the univariate marginals are proper distributions        
        self.prior = {
            "q": {"sample": None},
            "theta": {"sample": None},
            "proper": True}
        
        self.post = {
            "q": {"sample": None},
            "theta": {"sample": None},
            "proper": True}
            
        self.pdf = pdf
        
            
    def get_samples(
            self,
            typ,
            init,
            prop_sd,
            p,
            data=None,
            save=False):
        """
        MCMC algorithm for sampling theta and q with log(sigma) parametrisation
        from prior or posterior.
        -----------------------------------------------------------------------
        typ:     "prior" or "post".
        init:    Initialisation for MCMC algorithm.
        prop_sd: Proposal normal standard deviations for MCMC algorithm.
        p:       [p_1, p_2, p_3].
        data:    GEVData for posterior.
        """
        
        # Number of iterations
        N = 501000
        
        # Length of burn-in
        burnin = 1000
        
        if typ == "post":
            self.post["data"] = data
            
            # Transformation M -> M*
            prior = lambda X: self.pdf(X) * data.theta_annual_det(X)
            
            target = lambda X: util.logpost(X, prior, data)
        else:
            target = self.pdf
            
        # Transformation sigma -> log(sigma)
        new_target = util.log_transform(target, util.sig_trans, lambda X: X[1])
        
        mcmc_obj = util.MCMCSample(
            new_target,
            init,
            prop_sd,
            N,
            burnin)
        
        mcmc_obj.results(
            para_names=[
                r"$\tilde{\mu}$",
                r"$\log\tilde{\sigma}$",
                r"$\tilde{\xi}$"],
            colour=self.colour,
            save=save,
            save_name="%s-%s-%s-%s" % (
                self.inst_name,
                *self.hyperpara,
                typ))
        
        if typ == "post":
            # Transformation M* -> M
            sample_theta = np.array([data.theta_annual(X) for X in mcmc_obj.sample])
        else:
            sample_theta = mcmc_obj.sample
            
        getattr(self, typ)["theta"]["sample"] = sample_theta
            
        # Transformation log(sigma) -> sigma
        sample_theta[:, 1] = np.exp(sample_theta[:, 1])
        
        sample_q = np.array([
            util.quantile(theta, 1.0 - p)
            for theta in sample_theta])
        
        getattr(self, typ)["q"]["sample"] = sample_q
        
        getattr(self, typ)["mcmc"] = mcmc_obj
        
        
    def set_marginals(self, bw_method=None):
        """
        Generates marginal PDFs of theta and (q1, q2, q3), for prior and
        posterior distributions.
        If there exist samples, uses KDE.
        For priors on (q1, q2, q3), uses openturns.
        -----------------------------------------------------------------------
        bw_method: bandwidth method (see SciPy documentation)
        """
                
        # Log density
        def kde(sample):
            k = gaussian_kde(np.transpose(sample), bw_method=bw_method)
            return lambda X: k.logpdf(np.array(X))[0]
        
        for para in ["theta", "q"]:
            for typ in ["prior", "post"]:
                sample = getattr(self, typ)[para]["sample"]
                
                if sample is None:
                    getattr(self, typ)[para]["marginal"] = [
                        None
                        for I in util.marg_1_2]
                    continue
                
                getattr(self, typ)[para]["marginal"] = [
                    kde(sample[:, I])
                    for I in util.marg_1_2]
                
        if self.hyperpara[0] == 3:
            if self.hyperpara[1] == "i":
                qu_diff_dist = [
                    TruncatedDistribution(
                        Normal(self.para[i, 0], self.para[i, 1]),
                        0.0,
                        TruncatedDistribution.LOWER)
                    for i in range(3)]
                qu_dist = [
                    qu_diff_dist[0],
                    qu_diff_dist[0] + qu_diff_dist[1],
                    qu_diff_dist[0] + qu_diff_dist[1] + qu_diff_dist[2]]
                
                self.prior["q"]["marginal"][:3] = [
                    qu_dist[i].computeLogPDF
                    for i in range(3)]
            elif self.hyperpara[1] == "me":
                self.prior["q"]["marginal"][:3] = [
                    TruncatedDistribution(
                       Normal(self.para[i, 0], self.para[i, 1]),
                       0.0,
                       TruncatedDistribution.LOWER).computeLogPDF
                   for i in range(3)]
    
            
class PriorQ(PriorTheta):
    def __init__(self, p, hyperpara, para, inst_name=""):
        """
        Prior from specification of quantiles or quantile differences.
        -----------------------------------------------------------------------
        p:         probabilities [p_1, p_2, p_3] (np.ndarray).
        hyperpara: list [x, y]:
                   x: 2 (k = 2) or 3 (k = 3)
                   y: "i" (independent copula) or
                      "me" (maximum entropy copula).
        para:      2d numpy array, whose i-th row are parameters
                   [parent mean, parent standard deviation]
                   of the i-th marginal.
        -----------------------------------------------------------------------
        Returns: PriorTheta.
        """
        
        self.hyperpara = hyperpara
        self.para = para
        
        s = -np.log(-np.log(1.0 - p))
        
        k, cop = hyperpara
        
        # Sets colour
        colour = [
            None,
            None,
            {
                "i": 2,
                "me": 3},
            {
                "i": 0,
                "me": 1}][k][cop]
        
        if cop == "i":
            if k == 3:
                def f(X):
                    q1, q2, q3 = X
                    
                    Y = np.array([q1, q2 - q1, q3 - q2])
                    
                    if np.any(Y <= 0.0):
                        return None
                    
                    return -0.5 * np.sum(((Y - para[:, 0]) / para[:, 1]) ** 2)
            elif k == 2:
                def f(X):
                    sigma, q1, q2 = X
                    
                    if sigma <= 0:
                        return None
                    
                    Y = np.array([q1, q2 - q1])
                    
                    if np.any(Y <= 0.0):
                        return None
                    
                    a = -0.5 * np.sum(((Y - para[:, 0]) / para[:, 1]) ** 2)
                    
                    return a - np.log(sigma)
        elif cop == "me":
            q_marg = [None for _ in range(k)]
            
            for i in range(k):
                dist = Normal(para[i, 0], para[i, 1])
                q_marg[i] = TruncatedDistribution(
                    dist,
                    0.0,
                    TruncatedDistribution.LOWER)
            
            ot = MaximumEntropyOrderStatisticsDistribution(q_marg)
            
            if k == 3:
                def f(X):
                    q1, q2, q3 = X
                    
                    if np.any(np.array([q1, q2 - q1, q3 - q2]) <= 0.0):
                        return None
                    
                    Y = ot.computePDF(X)
                    
                    if Y <= 0:
                        return None
                    
                    return np.log(Y)
            elif k == 2:
                def f(X):
                    sigma, q1, q2 = X
                    
                    if sigma <= 0 or q1 <= 0.0 or q2 <= q1:
                        return None
                    
                    Y = ot.computePDF([q1, q2])
                    
                    if Y <= 0:
                        return None
                    
                    return np.log(Y) - np.log(sigma)
        
        if k == 3:
            # Transformation (mu, theta, xi) -> (q1, q2, q3)
            def g(X):
                mu, sigma, xi = X
                
                if sigma <= 0:
                    return None
                
                # When xi is close enough to 0, we consider it equal to 0
                if abs(xi) < 1e-300:
                    q = mu + sigma * s
                else:
                    q = mu + sigma * (np.exp(xi * s) - 1.0) / xi
            
                if q[0] < 0.0:
                    return None
                return q
        
        
            # Log of determinant of g
            def g_det(X):
                mu, sigma, xi = X
                
                if abs(xi) < 1e-300:
                    return np.log(sigma)
                    
                e = np.exp(s * xi)
                
                sm = [
                    s[i] * e[i] * (e[(i + 2) % 3] - e[(i + 1) % 3])
                    for i in range(3)]
                    
                return np.log(sigma) + np.log(sum(sm)) - np.log(xi ** 2.0)
        elif k == 2:
            # Transformation (mu, sigma, xi) -> (sigma, q1, q2)
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
        
        super().__init__(
            util.log_transform(f, g, g_det),
            colour=colour,
            inst_name=inst_name)
        
        if k == 2:
            self.prior["proper"] = False
    

def all_priors(
        p,
        qu,
        var,
        name=""):
    """
    Returns a list of all four priors. See Section 3.4.
    ---------------------------------------------------------------------------
    p:    [p_1, p_2, p_3] (1d numpy array).
    qu:   Estimate of (q_1, q_2, q_3).
    var:  Variances of quantile differences.
    name: Label.
    ---------------------------------------------------------------------------
    Returns: List of all four priors.
    """
    
    # Means of quantile differences
    qu_diff_m = np.array([qu[0], qu[1] - qu[0], qu[2] - qu[1]])
    
    # Means and variances of quantile differences
    qu_diff_mv = np.transpose(np.array([qu_diff_m, var]))
    
    # Parameters for quantile differences
    para_qu_diff = np.array([util.tn_para(*row) for row in qu_diff_mv])
    
    # Parameters for quantiles
    para_qu = util.para_for_quantiles(para_qu_diff)
    
    return np.array([
            PriorQ(p, [3, "i"], para_qu_diff, name),
            PriorQ(p[:2], [2, "i"], para_qu_diff[:2], name),
            PriorQ(p, [3, "me"], para_qu, name),
            PriorQ(p[:2], [2, "me"], para_qu[:2], name)])