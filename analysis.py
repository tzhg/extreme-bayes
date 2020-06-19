# Code for report version 7

from time import time
from csv import reader

import numpy as np

import util, priors

# Saves charts in folder "plots", MCMC tables in folder "mcmc-tables"
save_all = True

# --------------------------------------------------------------------------- #
# 1. Simulation study: Poisson point process (ppp)
# =========================================================================== #

time0 = time()

study_name = "ppp"

# Simulating data
# ===============

ppp_theta = [25.0, 5.0, 0.2]
ppp_M = 54
ppp_n_exceed = 86

ppp_data = util.poisson_point_process(
    ppp_theta,
    M=ppp_M,
    no_exceed=ppp_n_exceed,
    load=True,
    save=False,
    filename=study_name)

ppp_data.draw(save=save_all, save_name=study_name)

# Constructing priors
# ===================

p = [0.1, 0.01, 0.001]
var = 100

qu = np.array([util.quantile(ppp_theta, 1.0 - x) for x in p])
print([round(x, 3) for x in qu])

qu_diff = np.array([qu[0], qu[1] - qu[0], qu[2] - qu[1]])
print([round(x, 3) for x in qu_diff])

para_qu_diff = np.vstack((qu_diff ** 2 / var, qu_diff / var)).T

ppp_pi = priors.all_priors(p, para_qu_diff)

ppp_pi[1].details()

# MCMC sampling
# =============

# Set this to i to only sample from the i-th prior.
# Useful for tuning parameters.
isolate = None

for i in range(len(ppp_pi)):
    if isolate is not None and i is not isolate:
        continue
    
    ppp_pi[i].mcmc(
        "prior",
        [
            [25.0, 0.2, 0.25],
            [25.0, 0.2, 0.25],
            None,
            [25.0, 0.2, 0.25, 1.0],
            [25.0, 0.2, 0.25],
            [25.0, 0.2, 0.25]][i],
        [
            [40.0, 1, 0.2],
            [40.0, 1, 0.3],
            None,
            [40.0, 1, 0.2, 0.0],
            [100.0, 2.5, 0.8],
            [250.0, 2, 0.6]][i],
        save=save_all,
        save_name=study_name)
    
    ppp_pi[i].mcmc(
        "post",
        [
            [26.0, 1.8, 0.15],
            [26.0, 1.8, 0.15],
            [26.0, 1.8, 0.15],
            [26.0, 1.8, 0.15, 1.0],
            [26.0, 1.8, 0.15],
            [26.0, 1.8, 0.15]][i],
        [
            [4.0, 0.5, 0.2],
            [4.0, 0.5, 0.3],
            [4.0, 0.5, 0.4],
            [4.0, 0.5, 0.2, 0.0],
            [4.0, 0.5, 0.4],
            [4.0, 0.5, 0.5]][i],
        data=ppp_data,
        save=save_all,
        save_name=study_name)

print(np.mean(ppp_pi[3].post["mcmc"].sample[:, 3]))

# Visualisation
# =============

ppp_pi[0].set_marginals()

util.draw_list_priors_marginals(
    ppp_pi,
    support={
        "q": [[0.0, 90.0], [0.0, 210.0], [0.0, 350.0]],
        "theta": [[-120.0, 100.0], [-6.0, 8.0], [-1, 1.4]]},
    save=save_all,
    save_name=study_name)

# Return level plot
# =================

# Analytic return level:

X = np.logspace(0.0001, 3.0, 20)
Y = [util.quantile(ppp_theta, 1.0 - 1.0 / x) for x in X]
ppp_true_rl = [X, Y]

# Empirical return level:

# Number of samples
n = 1000

ppp_emp_rl = util.poisson_point_process(
    ppp_theta,
    M=ppp_M * n,
    no_exceed=ppp_n_exceed * n).qq()

for i, prior in enumerate(ppp_pi):
    util.plot_return_level(
        prior.post["theta"]["sample"],
        ppp_data,
        prior.colour,
        true_rl=ppp_true_rl,
        emp_rl=ppp_emp_rl,
        ylim=310,
        save=save_all,
        save_name="%s-post-%s" % (study_name, i))
    
print("%s: %f min" % (study_name, (time() - time0) / 60.0))

# %%------------------------------------------------------------------------- #
# 2. Simulation study: pseudo-data (pd)
# =========================================================================== #

time0 = time()

study_name = "pd"

# Simulating data
# ===============

pd_theta = [17, 1.8, 0.28]
pd_n_exceed = 86
pd_M = 54

pd_data = util.sim_gev(
    pd_theta,
    pd_M,
    pd_n_exceed,
    load=True,
    save=False,
    filename=study_name)

pd_data.draw(save=save_all, save_name=study_name)

print(pd_data.u)

# Constructing priors
# ===================

p = [0.1, 0.01, 0.001]
pd_para = [[38.9, 0.67], [7.1, 0.16], [47.0, 0.39]]

pd_pi = priors.all_priors(p, pd_para)

pd_pi[1].details()

# MCMC sampling
# =============

# Set this to i to only sample from the i-th prior.
# Useful for tuning parameters.
isolate = None

for i in range(len(pd_pi)):
    if isolate is not None and i is not isolate:
        continue
    
    pd_pi[i].mcmc(
        "prior",
        [
            [50.0, 1.5, 0.5],
            [50.0, 1.5, 0.5],
            None,
            [50.0, 1.5, 0.5, 1],
            [50.0, 1.5, 0.5],
            [50.0, 1.5, 0.5]][i],
        [
            [40, 1.2, 0.3],
            [40, 0.6, 0.2],
            None,
            [40, 0.6, 0.2, 0],
            [150, 3, 0.6],
            [250, 3, 0.6]][i],
        save=save_all,
        save_name=study_name)
    
    pd_pi[i].mcmc(
        "post",
        [
            [43.0, 2.2, 0.25],
            [43.0, 2.2, 0.25],
            [43.0, 2.2, 0.25],
            [43.0, 2.2, 0.25, 1],
            [43.0, 2.2, 0.25],
            [43.0, 2.2, 0.25]][i],
        [
            [4, 0.5, 0.3],
            [4, 0.5, 0.3],
            [4, 0.5, 0.3],
            [5, 0.5, 0.3, 0],
            [4, 0.5, 0.4],
            [4, 0.5, 0.4]][i],
        data=pd_data,
        save=save_all,
        save_name=study_name)

print(np.mean(pd_pi[3].post["mcmc"].sample[:, 3]))

# Visualisation
# =============
#%%

pd_pi[0].set_marginals()

util.draw_list_priors_marginals(
    pd_pi,
    support={
        "q": [[0.0, 100.0], [0.0, 300.0], [0.0, 500.0]],
        "theta": [[-30.0, 100.0], [-10, 30.0], [-1, 2]]},
    save=save_all,
    save_name=study_name)
#%%
# Return level plot
# =================

# Analytic return level:

X = np.logspace(0.0001, 3.0, 20)
Y = [util.quantile(pd_theta, (1.0 - 1.0 / x) ** (1 / 365.0)) for x in X]
pd_true_rl = [X, Y]

# Empirical return level:

# Number of samples
n = 1000

pd_emp_rl = util.sim_gev(
    M=pd_M * n,
    no_exceed=pd_n_exceed * n,
    para=pd_theta).qq()

for i, prior in enumerate(pd_pi):
    util.plot_return_level(     
        prior.post["theta"]["sample"],
        pd_data,
        prior.colour,
        true_rl=pd_true_rl,
        emp_rl=pd_emp_rl,
        ylim=700,
        save=save_all,
        save_name="%s-post-%s" % (study_name, i))

# Median and 0.95-quantile of quantile differences
# ================================================

for j in range(len(pd_pi)): 
    val_qu = np.array([
        [
            util.quantile(theta, 1.0 - pr)
            for pr in p]
        for theta in pd_pi[j].prior["theta"]["sample"]])
    
    val_qu_diff = np.transpose(np.array([
        val_qu[:, 0],
        val_qu[:, 1] - val_qu[:, 0],
        val_qu[:, 2] - val_qu[:, 1]]))
    
    st = [
        "(%s)" % ", ".join([
            util.latex_f(x)
            for x in [
                    np.quantile(val_qu_diff[:, i], 0.5),
                    np.quantile(val_qu_diff[:, i], 0.9)]])
        for i in range(3)]
    
    print(r"$\pi_{\theta}^{\text{%s}}$ &%s \\" % (
        pd_pi[j].name,
        " &".join(st)))
#%%
# Varying threshold
# =================

# List of numbers of exceedances
vt_list = [int(x) for x in np.linspace(10, 200, 21)]

vt_data = [
    util.sim_gev(
        pd_theta,
        pd_M,
        x,
        load=True,
        filename=study_name)
    for x in vt_list]
    
util.vary_threshold(
    vt_list,
    [
        util.PriorQD(p, pd_para)
        for x in vt_list],
    inits=[[44.0, 7.0, 0.05]] * 21,
    prop_sds=(
        [[8, 0.4, 0.1]] * 2
        + [[6, 0.4, 0.1]] * 6
        + [[6, 0.4, 0.12]] * 10
        + [[5, 0.4, 0.12]] * 3),
    data=[
        util.sim_gev(
            pd_theta,
            pd_M,
            x,
            load=True,
            save=False,
            filename=study_name)
        for x in vt_list],
    emp_rl=ppp_emp_rl,
    save=save_all,
    save_name=study_name)

print("%s: %f min" % (study_name, (time() - time0) / 60.0))

# %%------------------------------------------------------------------------- #
# 3. Real data: wind speed (ws)
# =========================================================================== #

time0 = time()

study_name = "ws"

# Simulating data
# ===============

ws_data_raw = util.load_data(study_name)
                
ws_M = 0
ws_u = 0

ws_data = util.GEVData(ws_u, ws_data_raw, ws_M)

ws_data.draw(save=save_all, save_name=study_name)

print("%s: %f min" % (study_name, (time() - time0) / 60.0))