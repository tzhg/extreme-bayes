from random import random, gauss, shuffle
from time import time
from math import floor, ceil
from csv import reader

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, genextreme


# 1 and 2-dim marginal indices
marg_1_2 = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]

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

class DiscreteDist:
    def __init__(
            self,
            dim,
            pdf,
            steps,
            support,
            pdf_is_disc=False,
            normalise=False,
            log=False):
        """
        Discretises PDF for graphing purposes.
        dim:             number of dimensions of distribution (int)
        pdf:             (log) PDF which inputs a list of floats and outputs a
                         float (function) or discrete PDF (numpy ndarray)
        steps:           steps in the discretisation (int)
        support:         finite support (list of [float, float])
        pdf_is_discrete: True if pdf is already discrete (bool)
        log:             True if pdf is log of true pdf (bool)
        Returns:         DiscreteDist with attributes:
            X: grid of points in support
            Y: PDF of grid
        """
        self.dim = dim
        self.pdf = pdf
        self.steps = steps
        self.support = support
        
        self.X = np.array([
            np.linspace(*support[i], steps[i], endpoint=False)
            for i in range(dim)])
        
        if pdf_is_disc: 
            Y = pdf
        else:
            Y = np.zeros(steps)
            
            with np.nditer(
                    Y,
                    flags=["multi_index"],
                    op_flags=["readwrite"]) as it:
                for y in it:
                    X = [
                        self.X[i][it.multi_index[i]]
                        for i in range(dim)]
                    
                    y[...] = pdf(X)
            it.close()

        if log:
            Y = np.exp(Y)
        
        if normalise:
            extent = np.prod([su[1] - su[0] for su in support])
            self.Y = Y / (extent * np.mean(Y))
        else:
            self.Y = Y
        
        # All 1 and 2-dim marginals
        if dim == 3:
            self.marginal = [
                self.get_marginal(I)
                for I in marg_1_2]
            
    def get_marginal(self, I):
        """
        Returns a marginal distribution as a DiscreteDist
        I: list of indices for marginal distribution
        """
        
        l = list(range(self.dim))
    
        for i in sorted(I, reverse=True):
            del l[i]
            
        new_Y = self.Y
        
        for i in sorted(l, reverse=True):
            new_Y = (
                self.support[i][1]
                - self.support[i][0]) * np.mean(new_Y, i)
        
        return DiscreteDist(
            len(I),
            new_Y,
            [self.steps[i] for i in I],
            [self.support[i] for i in I],
            pdf_is_disc=True)
        

def log_transform(log_pdf, g_inv, log_det):
    """
    Transformation of log PDFs
    g_inv:   inverse of pdf transformation (function)
    log_det: log of absolute value of determinant of Jacobian of g_inv
             (function)
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
    Transformation (mu, log(sigma), xi) -> (mu, sigma, xi)
    X: (mu, log(sigma), xi)
    """
    Y = X.copy()
    Y[1] = np.exp(Y[1])
    return Y


def latex_f(x):
    """
    Formats numbers for LateX
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
            _burnin,
            full_cond=None):
        """
        Metropolis-Within-Gibbs algorithm.
        For each variable, samples from full conditionals if given,
        otherwise samples from symmetric Normal proposal distributions.
        
        pdf:        log target density
        init:       vector of initial values
        prop_sd:    list of sd's of Normal proposal distributions
        N:          number of iterations
        burnin:     number of iterations to throw away
        full_cond:  list of full conditionals of each variable
        """
        
        std = time()
        
        self.no_para = len(init)
        self.N = N
        self.burnin = _burnin
        self.prop_sd = prop_sd
        self.init = init
        
        theta = init.copy()
        
        y = pdf(theta)
    
        acc = [0.0 for _ in range(self.no_para)]
        
        sample = []
      
        for i in range(N):
            for j in range(self.no_para):
                theta_star = theta.copy()
    
                if full_cond is None or full_cond[j] is None:
                    # If no full conditional
                    theta_star[j] = gauss(theta_star[j], prop_sd[j])
                    
                    y_star = pdf(theta_star)
                    
                    accept = y_star - y > np.log(random())
                else:
                    compl = theta.copy()
                    
                    del compl[j]
                    
                    theta_star[j] = full_cond[j](compl)
                    
                    y_star = pdf(theta_star)
                    
                    accept = True
                    
                if accept:
                    theta = theta_star
                    y = y_star
        
                    if i >= _burnin:
                        acc[j] += 1
            
            if i >= _burnin:
                sample.append(theta)
        
        sample = np.array(sample)
            
        acc_rate = np.array(acc) / (N - _burnin)
                
        print("MCMC: %d samples %s s (%s)" % (
            N,
            latex_f(time() - std),
            ", ".join([latex_f(x) for x in acc_rate])))
         
        self.sample = sample
        self.acc_rate = acc_rate
        self.mean = list(np.mean(sample, 0))
   
    def plot_trace(self, para_names=["", "", ""], save=False, save_name=""):
        """
        Plots trace plot(s) of the variables

        """
        n = 1000
        
        fig, ax = plt.subplots(nrows=1, ncols=self.no_para)
        k = floor((self.N - self.burnin) / n)
        for i in range(self.no_para):
            if self.no_para == 1:
                cell = ax
            else:
                cell = ax[i]
            
            cell.plot(
                np.array(range(n)) * k,
                [np.transpose(self.sample)[i][j * k] for j in range(n)],
                pal[0])
            
            cell.grid()
            cell.set(ylabel=para_names[i])
        fig.set_size_inches(8, 2)
        fig.tight_layout()
        if save:
            plt.savefig("plots/%s.pdf" % save_name, bbox_inches="tight")
        
        plt.show()
        
        
    def sample_hist(
        self,
        para_names=["", "", ""],
        save=False,
        save_name=""):
        """
        Draws histogram of the variables
        """
        nbins = ceil(2.0 * len(self.sample) ** (1.0 / 3.0))
        dim = len(self.sample[0])
        fig, ax = plt.subplots(nrows=1, ncols=dim)
        for i in range(dim):
            if dim == 1:
                cell = ax
            else:
                cell = ax[i]
            cell.hist(
                self.sample[:, i],
                density=True,
                bins=nbins)
            
            cell.grid()
            cell.set(xlabel=para_names[i])
        fig.set_size_inches(8, 2)
        fig.tight_layout()
        if save:
            plt.savefig("plots/%s.pdf" % save_name, bbox_inches="tight")
        plt.show()


    def results(
            self,
            para_names=["", "", ""],
            disc_pdf=None,
            save=False,
            save_name=""):
        """
        Generates tables in report
        """
        
        fn = [
            "../plots/%s-%s.pdf" % (save_name, _type)
            for _type in ["trace", "hist"]]
        
        st = "$%s$,\quad$%s$,\quad$%s$" % (
            "\mu^{\\text{new}}\sim{\cal N}(\mu^{\\text{old}}, %s^2)" % latex_f(self.prop_sd[0]),
            "\log\sigma^{\\text{new}}\sim{\cal N}(\log\sigma^{\\text{old}}, %s^2)" % latex_f(self.prop_sd[1]),
            "\\xi^{\\text{new}}\sim{\cal N}(\\xi^{\\text{old}}, %s^2)" % latex_f(self.prop_sd[2]))
        
        st2 = "\metro{%s}{%s}{%s}{%s}{%s}{%s}{%s}\n" % (
            st,
            ",\quad".join([latex_f(x) for x in self.init]),
            "$%s$ (after burn-in of $%s$)" % (latex_f(self.N - self.burnin), latex_f(self.burnin)),
            *fn,
            ",\quad".join([latex_f(x) for x in self.acc_rate]),
            ",\quad".join([latex_f(x) for x in self.mean]))
        
        if save:
            with open("mcmc-tables/%s.txt" % save_name, "w") as text_file:
                print(st2, file=text_file)
        else:
            print(st2)
        
        self.plot_trace(
            para_names,
            save=save,
            save_name="%s-trace" % save_name)
        
        self.sample_hist(
            para_names,
            save=save,
            save_name="%s-hist" % save_name)
        

class GEVData:
    def __init__(self, u, x, _M):
        """
        Data to be modelled by Poisson point process model
        u:  threshold
        x:  observations
        _M: number of years of observations
        
        """
        self.u = u
        self.x = np.array(x)
        self.x_u = np.array([obs for obs in x if obs >= u])
        self.n = len(x)
        self.M = _M
    
    def draw(self, save=False, save_name=""):
        t = range(1, self.n + 1, 1)
        s = self.x
        
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        
        ax.vlines(t, [0], s)
        ax.hlines(
            self.u,
            0.0,
            1.0,
            transform=ax.get_yaxis_transform(),
            colors="r")
        
        fig.set_size_inches(6, 4)
    
        if save:
            fig.tight_layout()
            plt.savefig("plots\%s-data.pdf" % save_name, bbox_inches="tight")
        plt.show()
        
    def qq(self):
        """
        Returns empirical quantiles of the data
        """
        Y = [
            max(self.x[(365 * i):(365 * (i + 1) - 1)])
            for i in range(self.M)]
        
        Y.sort()
        
        X = 1.0 / (1.0 - (np.array(range(self.M)) + 1.0) / (self.M + 1.0))
        
        return np.array([X, Y])


def loglik(theta, data):
    """
    Log likelihood of model
    theta: (mu, sigma, xi)
    data:  GEVdata
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
    Log posterior of (mu, sigma, xi), up to constant
    X:         (mu, sigma, xi)
    log_prior: log prior of X
    data:      GEVdata
    """
    return log_prior(X) + loglik(X, data)


def quantile(X, _p):
    """
    p-quantile of GEV distribution (inverse of CDF)
    X:  parameters (mu, sigma, xi)
    _p: probability p
    """
    mu, sigma, xi = X
    
    _pp = -np.log(-np.log(_p))
    
    # When xi is close enough to 0, we consider it equal to 0
    if abs(xi) < 1e-300:
        return mu + sigma * _pp
        
    return mu + sigma * (np.exp(xi * _pp) - 1.0) / xi


def sample_ret_level(sample, x):
    """
    Estimates return level for sample at return period x
    sample: numpy array
    x:      return period (years)
    returns: [
        mean return level,
        0.05-quantile return level,
        0.95-quantile return level]
    """
    l = [quantile(theta, 1.0 - 1.0 / x) for theta in sample]
    return [np.mean(l, 0), np.quantile(l, 0.05), np.quantile(l, 0.95)]


def plot_return_level(
        sample,
        data,
        colour,
        true_rl=None,
        emp_rl=None,
        ylim=None,
        save=False,
        save_name=""):
    """
    Plots estimated return level
    sample:        numpy array
    data:          GEVdata
    colour:        colour
    specific_para: draws line corresponding to a specific theta
    emp_rl:        draws return level simulated from large sample
    """
    N = 20
    X = np.logspace(0.0001, 3.0, num=N)
    
    fig, ax = plt.subplots()
    
    ax.scatter(*data.qq(), s=10, color="k")
    
    if true_rl is not None:
        ax.plot(*true_rl, "k")
        
    if emp_rl is not None:
        n = int(np.shape(emp_rl)[1] / 1000)
        ax.plot(
            *emp_rl[:, n * (1000 - 600):n * (1000 - 1)],
            color="k",
            linestyle="dashed")
        
    Y = np.array([
        sample_ret_level(sample, x)
        for x in X])
    
    ax.plot(
        X,
        Y[:, 0],
        color=pal[colour])
    
    plt.fill_between(
        X,
        Y[:, 1],
        Y[:, 2],
        alpha=0.3,
        facecolor=pal[colour])
        
    ax.set_ylim([None, ylim])   
    ax.set(xlabel="Return period (years)", ylabel="Quantile")
    ax.grid()
    ax.set_xscale("log")
    
    fig.set_size_inches(5, 5)

    fig.tight_layout()
    if save:
        plt.savefig(
            "plots/%s-return-level.pdf" % save_name,
            bbox_inches="tight")
    plt.show()
    
para_names = {
    "q": [r"$q_1$", r"$q_2$", r"$q_3$"],
    "theta": [r"$\mu$", r"$\log(\sigma)$", r"$\xi$"]}
    
        
def draw_list_priors_marginals(
        list_priors,
        support,
        save=False,
        save_name=""):
    """
    Draws all 1 and 2-dim marginals of a list of priors
    list_priors: list of PriorTheta
    support: support of parameters in list_priors
    """
    
    for para in ["theta", "q"]:
        for i, I in enumerate(marg_1_2):
            disc =  [
                [
                    DiscreteDist(
                        len(I),
                        getattr(prior, m)[para]["marginal"][i],
                        [floor(1000.0 ** (1 / len(I))) for _ in I],
                        [support[para][j] for j in I],
                        log=True)
                    if getattr(prior, m)["proper"] else None
                    for m in ["prior", "post"]]
                for prior in list_priors]
            
            if len(I) == 1:
                mx = max([
                    y for prior in disc for m in prior if m is not None
                    for y in m.Y])
            
            fig, ax = plt.subplots(nrows=2, ncols=3)
            for row in range(2):
                for col in range(3):
                    cell = ax[row, col]
                    
                    prior = list_priors[3 * row + col]
                    
                    colr = {
                        "prior": pal_light[prior.colour],
                        "post": pal_dark[prior.colour]}
                    
                    for m in ["prior", "post"]:
                        if not getattr(prior, m)["proper"]:
                            continue
                        d = disc[3 * row + col][["prior", "post"].index(m)]
                        
                        if len(I) == 1:
                            # Univariate
                             cell.plot(
                                d.X[0],
                                d.Y,
                                colr[m],
                                label="%s %s" % (prior.name, m))               
                        else:
                            # Bivariate
                            grid = np.meshgrid(
                                *d.X,
                                indexing="ij")
                            cell.contour(
                                *grid,
                                d.Y,
                                colors=colr[m])
                            
                            # Diagonal shaded area
                            if para == "q":
                                diag = [
                                    max([support["q"][j][0] for j in I]),
                                    min([support["q"][j][1] for j in I])]
                                
                                cell.fill_between(
                                    diag,
                                    diag,
                                    diag[0] - 10,
                                    alpha=0.1,
                                    color="k")
                            
                            # Proxy used to detect lines for legend
                            cell.plot(
                                support[para][I[0]][0],
                                support[para][I[1]][0],
                                colr[m],
                                label="%s %s" % (prior.name, m))
                    
                    if len(I) == 1:
                        cell.set(
                            xlabel=para_names[para][I[0]],
                            xlim=tuple(support[para][I[0]]),
                            ylim=(0, mx * 1.05))
                    else:
                        cell.set(
                            xlabel=para_names[para][I[0]],
                            ylabel=para_names[para][I[1]],
                            xlim=tuple(support[para][I[0]]),
                            ylim=tuple(support[para][I[1]]))
                        
                    cell.legend(
                        ncol=2,
                        bbox_to_anchor=(0.5, 1),
                        loc="lower center",
                        frameon=False)
        
                    cell.grid()
            fig.set_size_inches(9, 4.5)
            fig.tight_layout()
            if save:
                plt.savefig(
                    "plots/%s-%s-%s-%s-marg.pdf" % (
                        save_name,
                        para,
                        floor(i / 3),
                        i - floor(i / 3) * 3),
                    bbox_inches="tight")
            plt.show()
    
    
def vary_threshold(
        no_exceeds,
        priors,
        inits,
        prop_sds,
        data,
        emp_rl,
        save=False,
        save_name=""):
    """
    Varies the threshold (or more correctly, the number of exceedances)
    no_exceeds: list of no of exceedances
    priors:     list of priors made with each no of exceedance
    data:       GEVdata
    emp_rl:     empirical quantiles from large sample
    """
    for i in range(len(no_exceeds)):
        priors[i].mcmc(
            "post",
            inits[i],
            prop_sds[i],
            data=data[i],
            save=False,
            save_name="ppp-vt")
    
    selected_years = [10.0, 100.0, 1000.0]
    selected_years_labels = ["$r = 10$", "$r = 10^2$", "$r = 10^3$"]
    
    sy_ret_level = np.array([
        [
            sample_ret_level(prior.post["theta"]["sample"], y)
            for prior in priors]
        for y in selected_years])
    
    n = int(np.shape(emp_rl)[1] / 1000)
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        cell = ax[i]
        cell.plot(no_exceeds, sy_ret_level[i, :, 0])
        cell.fill_between(
            no_exceeds,
            sy_ret_level[i, :, 1],
            sy_ret_level[i, :, 2],
            alpha=0.3,
            facecolor=pal[0])
        cell.axhline(
            y=emp_rl[1, n * (1000 - 10 ** (2 - i))],
            color="k",
            linestyle="dashed")
    
        cell.set(
            xlabel="Number of exceedances",
            ylabel="Return level for %s" % selected_years_labels[i])
        cell.grid()
    fig.set_size_inches(9, 3)
    
    fig.tight_layout()
    if save:
        plt.savefig("plots\\%s-vt.pdf" % save_name, bbox_inches="tight")
    plt.show()
    

def sim_gev(
        para,
        M,
        no_exceed,
        load=False,
        save=False,
        filename=""):
    """
    Simulates GEV distribution
    para:      parameters (mu, sigma, xi)
    M:         number of years of data
    no_exceed: number of exceedances
    load:      loads data from filename
    save:      saves data at filename
    """
    if load:
        load_data(filename)
    else:
        data = genextreme.rvs(
            -para[2],
            loc=para[0],
            scale=para[1], 
            size=M * 365)
        if save:
            np.savetxt("data\%s.csv" % filename, data, delimiter=",")
    
    return GEVData(np.sort(data)[::-1][no_exceed], data, M)
    
    
def poisson_point_process(
        para,
        M,
        no_exceed,
        load=False,
        save=False,
        filename=""):
    """
    Simulates Poisson point process
    para:      parameters (mu, sigma, xi)
    M:         number of years of data
    no_exceed: number of exceedances
    load:      loads data from filename
    save:      saves data at filename
    """
    mu, sigma, xi = para
    
    u = mu + sigma * ((no_exceed / M) ** (-xi) - 1) / xi
    
    if load:
        load_data(filename)
    else:
        data = genpareto.rvs(
            xi,
            loc=u,
            scale=sigma, 
            size=M * 365)
        if save:
            np.savetxt("data\%s.csv" % filename, data, delimiter=",")
    
    # Determines the indices of the exceedances
    shuffled_idx = [x for x in range(M * 365)]
    shuffle(shuffled_idx)
    
    trunc_data = [0 for _ in range(M * 365)]
    for i in range(no_exceed):
        trunc_data[shuffled_idx[i]] = data[i]
        
    return GEVData(u, trunc_data, M)


def load_data(filename):
    """
    Loads data in csv format, with one column and no header row

    """
    
    with open("data\%s.csv" % filename, newline="") as csvfile:
        csvreader = reader(csvfile, delimiter=",")
        data = [float(x[0]) for x in list(csvreader)]

    return data