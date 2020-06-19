# -*- coding: utf-8 -*-
"""
Created on Fri May  1 00:34:28 2020

@author: tonyz
"""

from itertools import product

#%%

#########################
from mpl_toolkits import mplot3d

points = [X for X in product(*prior[0].discrete_seq) if quantile(X, p[0]) < 0.0]

ax = plt.axes(projection='3d')
ax.scatter3D(*np.transpose(points), c=np.transpose(points)[2], cmap="Greens")

#####
#%%

from scipy.special import gamma

# Marginals of q_tilde
# q_tilde_marg = [
    # Gamma(38.9, 0.67, 0.0),
    # Gamma(7.1, 0.16, 0.0),
    # Gamma(47.0, 0.39, 0.0)]

mu = 40.0
sigma = 0.001

def f1(x):
    if x==0:
        x=0.001
    q = (mu + sigma * (np.exp(x * pp[0]) - 1.0) / x)
    return q ** (38.9 - 1.0) * np.exp(-0.67 * q)

def f2(x):
    if x==0:
        x=0.001
    q = (sigma * (np.exp(x * pp[1]) - np.exp(x * pp[0])) / x)
    return q ** (7.1 - 1.0) * np.exp(-0.16 * q)

def f3(x):
    if x==0:
        x=0.001
    q = (sigma * (np.exp(x * pp[2]) - np.exp(x * pp[1])) / x)
    return q ** (47.0 - 1.0) * np.exp(-0.39 * q)

def h(x):
    return (np.exp(x * pp[2]) - np.exp(x * pp[1])) / x

max_s = 1000

def integrand(sigma, h):
    x = 0.39 * sigma * h
    
    if np.exp(-x) == 0.0:
        return 0
    
    return (x / (0.39 * h)) ** (47.0 - 1.0) * np.exp(-x)

def integral(xi):
    if abs(xi) < 1e-10:
        xi = 0.00001
        
    hh = h(xi)
    
    if hh == 0:
        return 0
    
    seq = np.linspace(1e-100, max_s, 1000)
    
    s = max_s * np.mean([integrand(x, hh) for x in seq])
    
    return hh ** (47.0 - 1.0) * s

def marg(X):
    mu, sigma, xi = X
    if abs(xi) < 1e-10:
        xi = 0.00001
        
    hh = h(xi)
    
    if hh == 0:
        return 0
    
    return hh ** (-1.0) * 0.39 ** (-47.0) * gamma(47.0)

# ss = 50
# seqq = [
#     np.linspace(1e-400, 35.0, ss),
#     np.linspace(-0.2, 0.9, ss)]

# def ggg(f, beep=False):
#     Y = np.zeros([ss] * 2)
    
#     with np.nditer(
#             Y,
#             flags=["multi_index"],
#             op_flags=["readwrite"]) as it:
#        for y in it:
#            sigma = seqq[0][it.multi_index[0]]
#            xi = seqq[1][it.multi_index[1]]
#            y[...] = f([sigma, xi])
#     it.close()
    
#     pr = (35.0 - 1e-3)
    
#     s = pr * np.mean(Y, (0))
    
#     y = (
#         h(seqq[1]) ** (47.0 - 1.0)
#         * s)
#     return y

X = np.linspace(-0.2, 0.9, 20)

fig, ax = plt.subplots()
ax.plot(X, [integral(x) for x in X], color="blue")
ax.plot(X, [marg([mu, sigma, x]) for x in X], color="orange")
fig.show
#%%
def fd(X):
    mu, sigma, xi = X
    if xi==0:
        xi = 0.001
    q3 = sigma * (np.exp(xi * pp[2]) - np.exp(xi * pp[1])) / xi
    if q3 <= 0:
        return 0
    if np.exp(-0.39 * q3) == 0.0:
        return 0
    ret3 = q3 ** (47.0 - 1.0) * np.exp(-0.39 * q3)
    return ret3

test = Distribution(
    3,
    [[-30.0, 90.0], [1e-400, 35.0], [-0.1, 1.2]],
    steps=50,
    pdf=fd,
    para_names=theta_names)

margd = Distribution(
    3,
    [[-30.0, 90.0], [1e-400, 35.0], [-0.1, 1.2]],
    steps=50,
    pdf=marg,
    para_names=theta_names)

from scipy.integrate import nquad
 
X = np.linspace(-0.1, 1.2, 50)

st3 = time()
nnn = nquad(lambda x, y, z: prior[0].pdf([x, y, z]), [[-30.0, 90.0], [1e-400, 35.0], [-0.1, 1.2]])[0]
spint = np.array([
    nquad(lambda x, y: prior[0].pdf([x, y, z]), [[-30.0, 90.0], [1e-400, 35.0]])[0]
    for x in X]) / nnn
print(time() - st3)

plot_pdfs([prior[0], test, margd])
plot_pdfs([prior[0], test, margd], log=True)

#test.draw_pdf_contours(save)

margppp = prior[0].marginal([2])



fig, ax = plt.subplots()
ax.plot(X, np.array([fgf(x) for x in X]) / nnn, color="blue")
ax.plot(X, margppp.pdf_disc, color="orange")
fig.show







###############################################################################


# Bits removed

    for method in [0, 2]:
        file = "arrays/q%d-%d.npy" % (method, nsteps)
        if save_all:
            np.save(file, d_q[method].discretise())
        if load_all:
            q[method].discretise(np.load(file))
            
            

def mean(self):
    if self.pdf_disc is None:
        self.discretise()
    
    return [
        ((self.support[i][1] - self.support[i][0])
         * np.mean(self.marginal([i]).pdf_disc * self.discrete_seq[i]))
        for i in range(self.dim)]

####################################
# choice of u
####################################

if save_charts:
    for u in range(7):
        file = "arrays/u%d-%d.npy" % (u, u_steps)
        if save_all:
            np.save(file, u_post[u].discretise())
        if load_all:
            u_post[u].discretise(np.load(file))
         
    # PDFs
    if save_charts:
        save = "plots\\u-post.pdf"
    else:
        save = None
    plot_pdfs(u_post, pallette=pall_rainbow, save=save)
    
    # GEV parameter means
    u_post_means = [u_post[i].mean() for i in range(7)]
    
    u_post_means_rd = [[round(x, 3) for x in u_post[i].mean()] for i in range(7)]
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for i in range(3):
        cell = ax[i]
        cell.plot(u_list, np.transpose(u_post_means)[i])
        
        cell.set(xlabel="Threshold", ylabel=theta_names[i])
        cell.grid()
    fig.set_size_inches(*size_1x3)
    
    fig.tight_layout()
    if save_charts:
        plt.savefig("plots\\u-estimates.pdf", bbox_inches="tight")
    plt.show()
    
    
pal_rainbow = (
    "tab:red",
    "tab:orange",
    "tab:olive",
    "tab:green",
    "tab:cyan",
    "tab:blue",
    "tab:purple",
    "tab:pink",
    "tab:brown",
    "tab:gray")
    
###############################

# marginals

def plot_pdfs(
        dists,
        save=False,
        save_name="",
        same_axis=False,
        log=False,
        linestyle=None,
        palette=pal,
        para_names=["", "", ""]):
    """
    Plots marginals of multiple 3-dimensional distributions
    dists: list of DiscreteDist of dimension 3
    log:   True to plot log of dists
    """

    if palette is None:
        palette = pal
    n = len(dists)
    
    if linestyle is None:
        linestyle = ["solid"] * n
        
    if same_axis:
        fig, ax = plt.subplots()
        for d in range(n):
            for j in range(3):
                dist = dists[d].marginal([j])
                Y = dist.Y
                if log:
                    Y = np.log(Y)
                ax.plot(
                    dist.X[0],
                    Y,
                    palette[d],
                    linestyle=linestyle[d])
        
        ax.set(xlabel=", ".join(para_names))
        ax.grid()
        
        fig.set_size_inches(4.5, 2.5)
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        for k in range(2):
            for l in range(2):
                cell = ax[k, l]
                i = 2 * k + l
                if i == 3:
                    cell.axis("off")
                    break
                for d in range(n):
                    dist = dists[d].marginal([i])
                    Y = dist.Y
                    if log:
                        Y = np.log(Y)
                    cell.plot(
                        dist.X[0],
                        Y,
                        palette[d],
                        linestyle=linestyle[d])
                
                cell.set(xlabel=para_names[i])
                cell.grid()
        fig.set_size_inches(4.5, 4.5)
    
    fig.tight_layout()
    if save:
        plt.savefig("plots/%s-pdf.pdf" % save_name, bbox_inches="tight")
    plt.show()


def draw_pdf_contours(
        dists,
        save=False,
        save_name="",
        log=False,
        para_names=["", "", ""]):
    """
    Draws contours of 2-dimensional marginals of 3-dimensional distributions
    dist: DiscreteDist
    """
    if dists[0].dim != 3:
        raise ValueError(
            "Can only draw PDF contours for 3 dimension distributions")
        
    n = len(dists)
          
    marg = [[0, 1], None, [0, 2], [1, 2]]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for k in range(2):
        for l in range(2):
            cell = ax[k, l]
            
            if marg[2 * k + l] is None:
                cell.axis("off")
                continue
            i, j = marg[2 * k + l]
            
            for m in range(n):
                d = dists[m].marginal([i, j])
    
                grid = np.meshgrid(
                    *d.X,
                    indexing="ij")
                
                Z = d.Y
                
                if log:
                    Z = np.log(Z)
                
                cell.contour(*grid, Z, colors=pal[m])
            
            if i == 0:
                cell.set(ylabel=para_names[j])
            if j == 2:
                cell.set(xlabel=para_names[i])

            cell.grid()
    fig.set_size_inches(4.5, 4.5)
    fig.tight_layout()
    if save:
        plt.savefig("plots/%s-contours.pdf" % save_name, bbox_inches="tight")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
        # draw_disc_marginals(
    #     [DiscreteDist(
    #             3,
    #             priors[i].q,
    #             nsteps,
    #             support_q[i],
    #             log=True).marginal
    #         for i in range(n) if priors[i].q is not None],
    #     save=save,
    #     save_name="%s-q" % save_name,
    #     para_names=quantile_names)
    
    # draw_disc_marginals(
    #     [DiscreteDist(
    #             3,
    #             priors[i].theta,
    #             nsteps,
    #             support_theta[i],
    #             log=True).marginal
    #         for i in range(n)],
    #     save=save,
    #     save_name="%s-theta" % save_name,
    #     para_names=theta_names)
    
    
    ############
    # SAS
    #############
    
    
    sas_prior = transform(
    q_prior(np.array([[38.9, 0.67], [7.1, 0.16], [4.0, 0.39]])),
    g,
    g_det)

sas_prior_mcmc = MCMCSample(
    sas_prior,
    [50, 10, 0],
    [
        lambda x: gauss(x, 7.5),
        lambda x: random() * x + x / 2.0,
        lambda x: gauss(x, 0.4)],
    100000,
    2000)
        
# Posterior:
    
sas_post_mcmc = MCMCSample(
    lambda X: logpost(X, sas_prior, sim_data),
    [50, 10, 0],
    [
        lambda x: gauss(x, 7.5),
        lambda x: random() * x + x / 2.0,
        lambda x: gauss(x, 0.4)],
    100000,
    2000)

spike_mcmc = MCMCSample(
    lambda X: logpost(X, sas_prior, sim_data),
    [50, 10, 0],
    [
        lambda x: gauss(x, 7.5),
        lambda x: random() * x + x / 2.0,
        lambda x: gauss(x, 0.4)],
    100000,
    2000,
    spike=True)

mix_post_sample = np.concatenate(
    (sas_post_mcmc.sample, spike_mcmc.sample),
    axis=0)

sample_hist(
    mix_post_sample,
    para_names["theta"],
    save=False,
    save_file="plots/sas-post-hist.pdf")

plot_return_level(
    [mix_post_sample],
    sim_data,
    save=True,
    save_name="sas")


#############################





def sample_GEV_para(sample, i):
    l = np.transpose(sample)[i]
    return [np.mean(l, 0), np.quantile(l, 0.05), np.quantile(l, 0.95)]