# -*- coding: utf-8 -*-

from random import random, gauss, shuffle
from time import time
from math import floor, ceil
from csv import reader

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, genextreme, norm
from scipy.optimize import root_scalar, minimize
from openturns import Normal, TruncatedDistribution


#=============================================================================#

# 1 and 2-dim marginal indices
marg_1_2 = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]

"""
In functions which save content, there will be a boolean parameter "save"
which determines whether the content is saved or not.
If save=True, the content will be saved in the folder with the relative
path given by save_path.
Plots are saved in a subfolder "plots",
Snippets of LateX are saved in a subfolder "latex-bits".
"""
save_path = "../report"

# Plot palette
pal = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan")

# To lighten/darken colours, I used http://scg.ar-ch.org/
# and chose colours at 20%

pal_light = (
    "#57a9e2",
    "#ffb574",
    "#5fd35f",
    "#e77c7c",
    "#c6aedc",
    "#bc8b81")

pal_dark = (
    "#103d5d",
    "#a74e00",
    "#165016",
    "#801718",
    "#613a84",
    "#4a2d27")
        

def log_transform(log_pdf, g_inv, log_det):
    """
    Transformation of log PDFs.
    ---------------------------------------------------------------------------
    g_inv:   Inverse of PDF transformation (function).
    log_det: Log of absolute value of determinant of Jacobian of g_inv
             (function).
    ---------------------------------------------------------------------------
    Returns: Transformed PDF (function).
    """
    
    def new_pdf(X):
        Y = g_inv(X)
        
        # Check if in domain of transformation
        if Y is None:
            return -np.inf
        
        Z = log_pdf(Y)
        
        if Z == None:
            return -np.inf

        return Z + log_det(X)
    
    return new_pdf


def sig_trans(X):
    """
    Transformation (mu, log(sigma), xi) -> (mu, sigma, xi).
    ---------------------------------------------------------------------------
    X: List [mu, log(sigma), xi].
    ---------------------------------------------------------------------------
    Returns: List [mu, sigma, xi].
    """
    Y = X.copy()
    Y[1] = np.exp(Y[1])
    return Y


def latex_f(x):
    """
    Formats numbers for LateX.
    ---------------------------------------------------------------------------
    x: Number.
    ---------------------------------------------------------------------------
    Returns: String.
    """
    s = f"{round(float(x), 3):,}"
    return s.rstrip("0").rstrip(".").replace(",", ",\!")


class MCMCSample:
    def __init__(
            self,
            pdf,
            init,
            prop_sd,
            N,
            _burnin):
        """
        Metropolis-Within-Gibbs algorithm.
        For each variable, samples from symmetric Normal proposal distributions
        with the given standard deviations.
        -----------------------------------------------------------------------
        pdf:       Log target density (function).
        init:      List of initial values.
        prop_sd:   List of sds of normal proposal distributions.
        N:         Number of iterations.
        _burnin:   Number of iterations to throw away at the start.
        """
        
        std = time()
        
        self.no_para = len(init)
        self.N = N
        self.burnin = _burnin
        self.prop_sd = prop_sd
        self.init = init
        
        theta = init.copy()
        
        y = pdf(theta)
    
        acc = [[] for _ in range(self.no_para)]
        
        sample = []
      
        i = 0
        while i < N:
            for j in range(self.no_para):
                theta_star = theta.copy()
    
                theta_star[j] = gauss(theta_star[j], prop_sd[j])
                    
                y_star = pdf(theta_star)
                
                accept = y_star - y > np.log(random())
                    
                if accept:
                    theta = theta_star
                    y = y_star
        
                    if i >= _burnin:
                        acc[j].append(1)
                else:
                    # theta = theta
                    # y = y
                    
                    if i >= _burnin:
                        acc[j].append(0)
            
            if i >= _burnin:
                sample.append(theta)
                
            i = i + 1
                
        sample = np.array(sample)
        
        acc_rate = [np.mean(acc[i]) for i in range(self.no_para)]
        
        print("%d samples %s s (%s)" % (
            N,
            round(time() - std),
            ", ".join(["%s" % round(x, 3) for x in acc_rate])))
         
        self.sample = sample
        self.acc_rate = acc_rate
        self.mean = list(np.mean(sample, 0))
   
    
    def plot_trace(
            self,
            para_names=["", "", ""],
            colour=pal[0],
            save=False,
            save_name=""):
        """
        Plots trace plot(s) of the variables.
        -----------------------------------------------------------------------
        para_names: Names of parameters for axis labels.
        colour:     Colour of plot.
        save_name:  Used to determine filename.
        """
        n = 1000
        
        fig, ax = plt.subplots(nrows=1, ncols=self.no_para, figsize=(7.5, 2))
        k = floor((self.N - self.burnin) / n)
        for i in range(self.no_para):
            if self.no_para == 1:
                cell = ax
            else:
                cell = ax[i]
            
            cell.plot(
                np.array(range(n)) * k,
                [np.transpose(self.sample)[i][j * k] for j in range(n)],
                colour)
            
            cell.grid()
            cell.set(ylabel=para_names[i])
        fig.tight_layout()
        if save:
            plt.savefig(
                "%s/plots/%s.pdf" % (save_path, save_name),
                bbox_inches="tight")
        
        plt.show()
        
        
    def sample_hist(
            self,
            para_names=["", "", ""],
            colour=pal[0],
            save=False,
            save_name=""):
        """
        Draws histogram of the variables.
        -----------------------------------------------------------------------
        para_names: Names of parameters for axis labels.
        colour:     Colour of plot.
        save_name:  Used to determine filename.
        """
        nbins = ceil(2.0 * len(self.sample) ** (1.0 / 3.0))
        dim = len(self.sample[0])
        fig, ax = plt.subplots(nrows=1, ncols=dim, figsize=(7.5, 2))
        for i in range(dim):
            if dim == 1:
                cell = ax
            else:
                cell = ax[i]
            cell.hist(
                self.sample[:, i],
                density=True,
                color=colour,
                bins=nbins)
            
            cell.grid()
            cell.set(xlabel=para_names[i])
        fig.tight_layout()
        if save:
            plt.savefig(
                "%s/plots/%s.pdf" % (save_path, save_name),
                bbox_inches="tight")
        plt.show()


    def results(
            self,
            para_names=["", "", ""],
            colour=0,
            save=False,
            save_name=""):
        """
        Generates MCMC tables in report.
        -----------------------------------------------------------------------
        para_names: Names of parameters for axis labels.
        save_name:  Used to determine filename.
        """
        
        self.plot_trace(
            para_names,
            colour=pal[colour],
            save=save,
            save_name="%s-trace" % save_name)
        
        self.sample_hist(
            para_names,
            colour=pal[colour],
            save=save,
            save_name="%s-hist" % save_name)
        
        if not save:
            return
        
        fn = [
            "plots/%s-%s.pdf" % (save_name, _type)
            for _type in ["trace", "hist"]]
        
        def para_str(star):
            if star:
                st = "^*"
            else:
                st = ""
            return [
                "\%s%s" % (par[2: -1], st)
                for i, par in enumerate(para_names)
                if self.prop_sd[i] is not None]
        
        st = r",\quad".join([
            "$%s\sim{\cal N}(%s,%s^2)$" % (
                para_str(True)[i],
                para_str(False)[i],
                latex_f(self.prop_sd[i]))
            for i in range(3)])
        
        st2 = "\metro{%s}{%s}{%s}{%s}{%s}{%s}{%s}\n" % (
            st,
            ",\quad".join(["$%s$" % latex_f(x) for x in self.init]),
            "$%s$ (after burn-in period of $%s$)" % (
                latex_f(self.N - self.burnin),
                latex_f(self.burnin)),
            *fn,
            ",\quad".join(["$%s$" % latex_f(x) for x in self.acc_rate]),
            ",\quad".join(["$%s$" % latex_f(x) for x in self.mean]))
    
        mcmc_table_file = "%s/latex-bits/%s.txt" % (save_path, save_name)
        with open(mcmc_table_file, "w") as text_file:
            print(st2, file=text_file)
        

class GEVData:
    def __init__(self, u, x, M=None, name=""):
        """
        Data to be modelled by Poisson point process model.
        -----------------------------------------------------------------------
        u:    Threshold (float).
        x:    List observations.
        M:    Number of blocks of data (float),
              default is number of exceedances.
        name: Label for data.
        -----------------------------------------------------------------------
        Returns: GEVData object
        """
        self.u = u
        self.x = np.array(x)
        self.name = name
        
        self.x_u = np.array([obs for obs in x if obs >= u])
        self.n = len(x)
        
        if M is None:
            M = len(self.x_u)
        self.M = M
        
        block_size = floor(self.n / M)
        
        self.block_max = [
            max(x[(block_size * i):(block_size * (i + 1) - 1)])
            for i in range(floor(M))]
        
        
    def fit_GEV(self, theta=None, save=False):
        """
        Fits GEV parameters to annual maxima of data using maximum
        likelihood (if theta is None) and draws Q-Q plot.
        -----------------------------------------------------------------------
        theta: Fitted [mu, sigma, xi].
        -----------------------------------------------------------------------
        Returns: Fitted [mu, sigma, xi].
        """
        
        block_max = [x for x in self.block_max if x > 0]
        block_max.sort()
        
        if theta is None:
            xi, mu, sigma = genextreme.fit(block_max)
        
            xi = -xi
            
            theta = [mu, sigma, xi]
            
        a = np.array(range(len(block_max))) + 1.0
        b = len(block_max) + 1.0
        
        emp_p = a / b
        emp_q = quantile(theta, emp_p)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.scatter(emp_q, block_max, s=10, color="k")
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="k")
        
        ax.set(
            xlabel="Theoretical quantiles",
            ylabel="Empirical quantiles")
        
        both = np.concatenate((emp_q, block_max))
        
        l = (max(both) - min(both)) / 10
        
        limits = [min(both) - l, max(both) + l]
        
        ax.grid(True)
        
        plt.axis("square")
        ax.axis([*limits, *limits])
        
        if save:
            fig.tight_layout()
            plt.savefig(
                "%s/plots/%s-qq.pdf" % (save_path, self.name),
                bbox_inches="tight")
        plt.show()
        
        return theta
        
    
    def draw(self, save=False):
        """
        Plots data.
        """
        t = range(1, self.n + 1, 1)
        s = self.x
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        ax.vlines(t, [0], s)
        ax.hlines(
            self.u,
            0.0,
            1.0,
            transform=ax.get_yaxis_transform(),
            colors="r")
        
        ax.set(
            xlabel="Day",
            ylabel="X")
    
        if save:
            fig.tight_layout()
            plt.savefig(
                "%s/plots/%s-data.pdf" % (save_path, self.name),
                bbox_inches="tight")
        plt.show()
        
        
    def optimal_M(self, xi):
        """
        Choses a suitable value of M.
        See Sharkey and J. A. Tawn 2017 eqs 12 and 13.
        -----------------------------------------------------------------------
        xi: Shape parameter xi.
        -----------------------------------------------------------------------
        Returns: GEVData object with suitable value of M.
        """
        l = (xi + 1) * np.log((2 * xi + 3) / (2 * xi + 1))
        m1 = (1 + 2 * xi + l) / (3 + 2 * xi - l)
        m2 = (2 * xi ** 2 + 13 * xi + 8) / (2 * xi ** 2 + 9 * xi + 8)
        
        opt_M = round(len(self.x_u) * (m1 + m2) / 2)
        
        return GEVData(
            u=self.u,
            x=self.x,
            M=opt_M,
            name=self.name)
    
    
    def set_obs_in_year(self, obs_in_year):
        """
        Sets the number of observations in a year and calculates the
        empirical quantiles of the annual maxima.
        -----------------------------------------------------------------------
        obs_in_year: the number of observations in a year.
        """
        self.obs_in_year = obs_in_year
        
        # Sets empirical quantiles of annual maxima
        
        M2 = int(floor(self.n / obs_in_year))
        
        Y = [
            max(self.x[(obs_in_year * i):(obs_in_year * (i + 1) - 1)])
            for i in range(M2)]
        
        Y.sort()
        
        X = 1.0 / (1.0 - ((np.arange(M2) + 1.0) / (M2 + 1.0)))
        
        self.emp_quant = np.array([X, Y])
    
    
    def theta_annual(self, X):
        """
        Reparametrises theta by changing M to the number of years of data.
        See Sharkey and J. A. Tawn 2017 eq. 5.
        Must have set obs_in_year first.
        -----------------------------------------------------------------------
        X: [mu, sigma, xi].
        -----------------------------------------------------------------------
        Returns: Reparametrised [mu, sigma, xi].
        """
    
        rat = self.n / (self.obs_in_year * self.M)
        return np.array([
            X[0] - (X[1] / X[2]) * (1 - rat ** -X[2]),
            X[1] * rat ** -X[2],
            X[2]])
    
    def theta_annual_det(self, X):
        """
        Reparametrises distribution on theta by changing block size to years.
        See Sharkey and J. A. Tawn 2017 eq. g.
        Must have set obs_in_year first.
        -----------------------------------------------------------------------
        X: [mu, sigma, xi].
        -----------------------------------------------------------------------
        Returns: Determinant of reparametrisation.
        """
    
        return ((self.obs_in_year * self.M) / self.n) ** -X[2]
        

def loglik(theta, data):
    """
    Log likelihood of model.
    ---------------------------------------------------------------------------
    theta: [mu, sigma, xi].
    data:  GEVData.
    ---------------------------------------------------------------------------
    Returns: Log likelihood.
    """
    mu, sigma, xi = theta
    
    if abs(xi) < 1e-300:
        pr = sum(-(data.x_u - mu) / sigma - np.log(sigma))
        return -data.M * np.exp(-(data.u - mu) / sigma) + pr
        
    a1 = 1.0 + xi * (data.u - mu) / sigma
    a2 = 1.0 + xi * (data.x_u - mu) / sigma
    
    # Support verification
    if a1 <= 0.0 or np.any(a2 <= 0.0):
        return -np.inf
    
    pr = sum(-((xi + 1.0) / xi) * np.log(a2) - np.log(sigma))
    
    return -data.M * a1 ** (-1.0 / xi) + pr


def logpost(X, log_prior, data):
    """
    Log posterior of (mu, sigma, xi), up to constant.
    ---------------------------------------------------------------------------
    X:         [mu, sigma, xi].
    log_prior: Log prior of theta (function).
    data:      GEVData.
    ---------------------------------------------------------------------------
    Returns: Log posterior.

    """
    return log_prior(X) + loglik(X, data)


def quantile(X, p):
    """
    p-quantile of GEV distribution with parameters X (inverse of CDF).
    ---------------------------------------------------------------------------
    X: [mu, sigma, xi].
    p: Probability p.
    ---------------------------------------------------------------------------
    Returns: p-quantile
    """
    mu, sigma, xi = X
    
    s = -np.log(-np.log(p))
    
    # When xi is close enough to 0, we consider it equal to 0
    if abs(xi) < 1e-300:
        return mu + sigma * s
        
    return mu + sigma * (np.exp(xi * s) - 1.0) / xi


def sample_ret_level(sample, x):
    """
    Estimates return level for sample at return period x.
    ---------------------------------------------------------------------------
    sample: Numpy 1d array of samples.
    x:      Return period (years).
    ---------------------------------------------------------------------------
    Returns: [
        mean return level,
        0.05-quantile return level,
        0.95-quantile return level].
    """
    l = [quantile(theta, 1.0 - 1.0 / x) for theta in sample]
    return [np.mean(l, 0), np.quantile(l, 0.05), np.quantile(l, 0.95)]


def discretise(
        dim,
        pdf,
        steps,
        support):
    """
    Discretises PDF for graphing purposes.
    ---------------------------------------------------------------------------
    dim:     Number of dimensions of distribution (int).
    pdf:     log PDF (function: list -> float).
    steps:   Steps in the discretisation (int).
    support: Finite support (list of [float, float]).
    ---------------------------------------------------------------------------
    Returns: {
        X: grid of points in support
        Y: PDF of X}
    """
    
    X = np.array([
        np.linspace(*support[i], steps[i], endpoint=False)
        for i in range(dim)])
    
    Y = np.zeros(steps)
    
    with np.nditer(
            Y,
            flags=["multi_index"],
            op_flags=["readwrite"]) as it:
        for y in it:
            _X = [
                X[i][it.multi_index[i]]
                for i in range(dim)]
            
            y[...] = pdf(_X)
    it.close()

    Y = np.exp(Y)
    
    return {"X": X, "Y": Y}
    
        
def draw_list_priors_marginals(
        list_priors,
        support,
        save=False):
    """
    Draws all 1- and 2-dim marginals of a list of priors.
    Does not save 2-dim marginals.
    ---------------------------------------------------------------------------
    list_priors: List of PriorTheta.
    support:     Support of parameters in list_priors:
            {
                theta: {
                    prior: [[mu1, mu2], [logsigma1, logsigma2], [xi1, xi2]],
                    post:  [[mu1, mu2], [logsigma1, logsigma2], [xi1, xi2]]},
                q: {
                    prior: [[q11, q12], [q21, q22], [q31, q32]],
                    post:  [[q11, q12], [q21, q22], [q31, q32]]}}
    """
        
    para_names = {
        "q": [r"$q_1$", r"$q_2$", r"$q_3$"],
        "theta": [r"$\mu$", r"$\log\sigma$", r"$\xi$"]}
         
    linestyle = {
        "prior": "dashed",
        "post": "solid"}
    
    for para in ["theta", "q"]:
        for i, I in enumerate(marg_1_2):
            # Disc is list of [X, f(X)] for each f = marginal PDF,
            # or None if marginal PDF does not exist
            disc = {
                m:[
                    discretise(
                        len(I),
                        getattr(prior, m)[para]["marginal"][i],
                        [floor(1000.0 ** (1 / len(I))) for _ in I],
                        [support[para][m][j] for j in I])
                    if getattr(prior, m)[para]["marginal"][i] is not None else None
                    for prior in list_priors]
                for m in ["prior", "post"]}
            
            if len(I) == 1:
                # Univariate
                fig, ax = plt.subplots(figsize=(9, 4.5))
                
                l = [x for m in ["prior", "post"] for x in support[para][m][I[0]]]
                
                ax.set(xlim=(min(l), max(l)))
    
                ax.set(
                    xlabel="%s" % para_names[para][I[0]],
                    ylabel="PDF")
                
                ax.grid(True)
                    
                for m in ["prior", "post"]:
                    for j, prior in enumerate(list_priors):
                        d = disc[m][j]
                        
                        # If there is no MCMC or analytic marginal
                        if d is None:
                            continue
                        
                        if len(I) == 1:
                            # Univariate
                            ax.plot(
                                d["X"][0],
                                d["Y"],
                                pal[prior.colour],
                                linestyle=linestyle[m])
                
                # Joining plots together
                fig.subplots_adjust(hspace=0)
                plt.setp(
                    [a.get_xticklabels() for a in fig.axes[:-1]],
                    visible=False)
                
                if save:
                    plt.savefig(
                        "%s/plots/%s-%s-%s-%s-marg.pdf" % (
                            save_path,
                            list_priors[0].inst_name,
                            para,
                            ["uni", "bi"][floor(i / 3)],
                            i - floor(i / 3) * 3),
                        bbox_inches="tight")
                plt.show()
                
            else:
                # Bivariate
                
                for m in ["prior", "post"]:
                    for j, prior in enumerate(list_priors):
                        d = disc[m][j]
                        
                        # If there is no MCMC or analytic marginal
                        if d is None:
                            continue
                    
                        fig, ax = plt.subplots()
                        
                        grid = np.meshgrid(
                            *d["X"],
                            indexing="ij")
                        ax.contour(
                            *grid,
                            d["Y"],
                            colors=pal[prior.colour])
                        
                        # Diagonal shaded area
                        if para == "q":
                            diag = [
                                max([support["q"][m][j][0] for j in I]),
                                min([support["q"][m][j][1] for j in I])]
                            
                            ax.fill_between(
                                diag,
                                diag,
                                diag[0] - 10,
                                alpha=0.1,
                                color="k")
                        
                        ax.set(
                            xlim=tuple(support[para][m][I[0]]),
                            ylim=tuple(support[para][m][I[1]]),
                            xlabel="%s" % para_names[para][I[0]],
                            ylabel="%s" % para_names[para][I[1]])
                    

def plot_return_level(
        list_priors,
        analytic_rl=None,
        save=False):
    """
    Plots estimated return level.
    ---------------------------------------------------------------------------
    list_priors: List of PriorTheta.
    analytic_rl: List [X, Y] of return level for some theta.
    """
    N = 50
    X = np.logspace(0.0001, 4.0, num=N)
    
    fig, ax = plt.subplots(figsize=(14, 9))
        
    for i, prior in enumerate(list_priors):		
        Y = np.array([
            sample_ret_level(prior.post["theta"]["sample"], x)
            for x in X])
        
        ax.plot(
            X,
            Y[:, 0],
            color=pal[prior.colour])
        
        ax.plot(
            X,
            Y[:, 1],
            color=pal[prior.colour],
            linestyle="dashed")
                
        ax.plot(
            X,
            Y[:, 2],
            color=pal[prior.colour],
            linestyle="dashed")
        
    if analytic_rl is not None:
        ax.plot(*analytic_rl, "k")
        
    ax.scatter(
        *list_priors[0].post["data"].emp_quant,
        s=10,
        color="k",
        zorder=10)
    
    plt.yscale("log")
        
    ax.set(xlabel="Return period (years)", ylabel="Return level")
    
    ax.grid()
    ax.set_xscale("log")

    fig.tight_layout()
    if save:
        plt.savefig(
            "%s/plots/%s-post-return-level.pdf" % (
                save_path,
                prior.inst_name),
            bbox_inches="tight")
    plt.show()


def vary_threshold(
        u_list,
        priors,
        inits,
        prop_sds,
        data,
        save=False,
        save_name=""):
    """
    Varies the threshold and produces plot of return level
    estimates for three years.
    ---------------------------------------------------------------------------
    u_list:   List of thresholds to test.
    priors:   List of identical priors.
    inits:    MCMC initialisations.
    prop_sds: MCMC normal proposal sds.
    data:     GEVData.
    """
    for i in range(len(u_list)):
        data2 = GEVData(u_list[i], data.x)
        data2.set_obs_in_year(data.obs_in_year)
        
        print(data2.u, len(data2.x_u))
        
        priors[i].get_samples(
            "post",
            inits[i],
            prop_sds[i],
            p=np.array([0.1, 0.01, 0.001]),
            data=data2,
            save=False)
        
    
    selected_years = [10.0, 100.0, 1000.0]
    selected_years_labels = ["$r = 10$", "$r = 10^2$", "$r = 10^3$"]
    
    sy_ret_level = np.array([
        [
            sample_ret_level(prior.post["theta"]["sample"], y)
            for prior in priors]
        for y in selected_years])
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    for i in range(3):
        cell = ax[i]
        cell.plot(u_list, sy_ret_level[i, :, 0])
        cell.fill_between(
            u_list,
            sy_ret_level[i, :, 1],
            sy_ret_level[i, :, 2],
            alpha=0.3,
            facecolor=pal[0])
    
        cell.set(
            xlabel="Threshold",
            ylabel="Return level for %s" % selected_years_labels[i])
        cell.grid()
    
    fig.tight_layout()
    if save:
        plt.savefig(
            "%s/plots/%s-vt.pdf" % (save_path, save_name),
            bbox_inches="tight")
    plt.show()
    
    
def poisson_point_process(
        para,
        n,
        no_exceed,
        load=False,
        save=False,
        filename="",
        name=None):
    """
    Simulates Poisson point process with 1 expected exceedance per block
    and number of blocks equal to number of exceedances.
    ---------------------------------------------------------------------------
    para:      [mu, sigma, xi].
    n:         Number of observations (int).
    no_exceed: Number of exceedances (int).
    load:      Loads data.
    save:      Saves data.
    filename:  Used to determine filename.
    name:      Label for the data.
    ---------------------------------------------------------------------------
    Returns: GEVData.
    """
    mu, sigma, xi = para
    
    if load:
        trunc_data = load_data(name)
    else:
        data = genpareto.rvs(
            xi,
            loc=mu,
            scale=sigma, 
            size=no_exceed)
    
        # Determines the indices of the exceedances
        shuffled_idx = [x for x in range(n)]
        shuffle(shuffled_idx)
        
        trunc_data = [0 for _ in range(n)]
        for i in range(no_exceed):
            trunc_data[shuffled_idx[i]] = data[i]
            
        if save:
            np.savetxt("data/%s.csv" % name, trunc_data, delimiter=",")
        
    return GEVData(mu, trunc_data, name=name)


def load_data(filename):
    """
    Loads data in csv format, with one column and no header row,
    from Data folder.
    ---------------------------------------------------------------------------
    filename: Filename of csv file.
    ---------------------------------------------------------------------------
    Returns:  List.
    """
    
    with open("data\%s.csv" % filename, newline="") as csvfile:
        csvreader = reader(csvfile, delimiter=",")
        data = [float(x[0]) for x in list(csvreader)]

    return data
    

def tn_para(m, V):
    """
    Converts mean and variance to truncated normal parameters.
    Only works for small enough variances.
    ---------------------------------------------------------------------------
    m: mean of TN distribution
    V: variance of TN distribution
    ---------------------------------------------------------------------------
    Returns: (parent mean, parent standard deviation)
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
    
    sol = root_scalar(g, x0=0, bracket=(-50, 20))
    
    alpha = sol.root

    sigma = m / (hazard(alpha) - alpha)
    mu = -alpha * sigma
    
    return (mu, sigma)

def para_for_quantiles(para_tn):
    """
    Given parameters for TN distribution of quantile differences,
    computes distribution of quantiles,
    and finds TN distributions which best approximate these distributions.
    ---------------------------------------------------------------------------
    para_tn: Parameters of distribution of quantile differences
             [parent mean, parent standard deviation]
    ---------------------------------------------------------------------------
    Returns: Parameters of distribution of quantiles
             [parent mean, parent standard deviation]
    """
    # TN
    
    mu_tn = para_tn[:, 0]
    sigma_tn = para_tn[:, 1]
    
    # Marginals of q_tilde
    q_tilde_marg_tn = [
        TruncatedDistribution(
            Normal(mu_tn[i], sigma_tn[i]),
            0.0,
            TruncatedDistribution.LOWER)
        for i in range(3)]
    
    # Marginals of q
    target_tn = [
        q_tilde_marg_tn[0],
        q_tilde_marg_tn[0] + q_tilde_marg_tn[1],
        q_tilde_marg_tn[0] + q_tilde_marg_tn[1] + q_tilde_marg_tn[2]]
    
    par_tn = [None for _ in range(3)]
    
    for i in range(3):
        # Truncated normal
        def KL_tn(par):
            mu, logsigma = par
            
            mu = float(mu)
            logsigma = float(logsigma)
            
            dist = TruncatedDistribution(
                Normal(mu, np.exp(logsigma)),
                0.0,
                TruncatedDistribution.LOWER)
        
            upper_bound = max(
                target_tn[i].getRange().getUpperBound()[0],
                dist.getRange().getUpperBound()[0])
            
            def integrand(x):
                pdf1 = dist.computePDF(x)
                pdf2 = target_tn[i].computePDF(x)
                pdf1 = max(pdf1, 1e-140)
                pdf2 = max(pdf2, 1e-140)
                return pdf1 * np.log(pdf1 / pdf2)
            
            X = np.linspace(0.0, upper_bound, 100)
            return upper_bound * np.mean([integrand(x) for x in X])
        
        res = minimize(KL_tn, (35.0, 1.0))
        
        mu, logsigma = res.x
            
        mu = float(mu)
        logsigma = float(logsigma)
        
        par_tn[i] = [mu, np.exp(logsigma)]
    
    return np.array(par_tn)