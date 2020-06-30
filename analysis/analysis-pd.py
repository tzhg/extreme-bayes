# -*- coding: utf-8 -*-
"""
Simulation study: pseudo-data (pd)
"""

from time import time

import numpy as np

import util, priors

#=============================================================================#

# Saves charts and LateX snippets at util.save_path
save_all = True

time0 = time()

study_name = "pd"

# Simulating data #-----------------------------------------------------------#
#=============================================================================#

pd_theta = [17, 1.8, 0.28]
pd_n_exceed = 86
pd_M = 54

pd_data = util.sim_gev(
    pd_theta,
    pd_M,
    pd_n_exceed,
    load=True,
    save=False,
    name=study_name)

pd_data.draw(save=save_all)

# Constructing priors #-------------------------------------------------------#
#=============================================================================#

p = [0.1, 0.01, 0.001]
pd_para = [[38.9, 0.67], [7.1, 0.16], [47.0, 0.39]]

pd_pi = priors.all_priors(
    p,
    0.5,
    para=pd_para,
    name=study_name,
    save=save_all)
#%%
# MCMC sampling #-------------------------------------------------------------#
#=============================================================================#

# Set this to i to only sample from the i-th prior.
# Useful for tuning parameters.
isolate = None

for i in range(len(pd_pi)):
    if isolate is not None and i is not isolate:
        continue
    
    pd_pi[i].mcmc(
        "prior",
        [50.0, 1.5, 0.4],
        [
            [40, 1.2, 0.3],
            None,
            None,
            [40, 0.6, 0.2],
            [150, 3, 0.6],
            [100, 1.5, 0.25]][i],
        save=save_all)
    
    pd_pi[i].mcmc(
        "post",
        [43.0, 2.2, 0.25],
        [
            [4, 0.5, 0.3],
            [4, 0.5, 0.3],
            [5, 0.5, 0.15],
            [4, 0.5, 0.3],
            [4, 0.5, 0.4],
            [4, 0.5, 0.25]][i],
        data=pd_data,
        save=save_all)
#%%
# Visualisation #-------------------------------------------------------------#
#=============================================================================#

util.draw_list_priors_marginals(
    pd_pi,
    support={
        "q": [[30.0, 100.0], [50.0, 250.0], [100.0, 800.0]],
        "theta": [[10.0, 80.0], [-3, 7.0], [-0.4, 0.9]]},
    save=save_all)

# Return level plot #---------------------------------------------------------#
#=============================================================================#

# Analytic return level:

X = np.logspace(0.0001, 3.0, 20)
Y = [util.quantile(pd_theta, (1.0 - 1.0 / x) ** (1 / 365.0)) for x in X]
pd_true_rl = [X, Y]

# Empirical return level:

# Number of samples
n = 5000

pd_emp_rl = util.sim_gev(
    M=pd_M * n,
    no_exceed=pd_n_exceed * n,
    para=pd_theta).emp_quant

for i, prior in enumerate(pd_pi):
    util.plot_return_level(     
        prior,
        true_rl=pd_true_rl,
        emp_rl=pd_emp_rl,
        ylim=650,
        save=save_all)

# Median and 0.9-quantile of quantile differences #---------------------------#
#=============================================================================#

st_list = []

for j in range(len(pd_pi)):
    if not pd_pi[j].prior["proper"]:
        continue
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
        "$(%s)$" % ", ".join([
            util.latex_f(x)
            for x in [
                    np.quantile(val_qu_diff[:, i], 0.5),
                    np.quantile(val_qu_diff[:, i], 0.9)]])
        for i in range(3)]
    
    st_list.append(r"$\pi_{\theta}^{\text{%s}}$ &%s \\" % (
        pd_pi[j].name,
        " &".join(st)))

if save_all:
    path = "%s/latex-bits/pd-prior-quantiles.txt" % util.save_path
    with open(path, "w") as text_file:
        print("\n".join(st_list), file=text_file)

#%%
# Varying threshold #---------------------------------------------------------#
#=============================================================================#

# List of numbers of exceedances
vt_list = [int(x) for x in np.linspace(10, 200, 21)]

vt_data = [
    util.sim_gev(
        pd_theta,
        pd_M,
        x,
        load=True,
        name=study_name)
    for x in vt_list]
    
util.vary_threshold(
    vt_list,
    [
        priors.PriorG3(p, pd_para, inst_name=study_name)
        for x in vt_list],
    inits=[[43.0, 2.2, 0.25]] * 21,
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
            name=study_name)
        for x in vt_list],
    emp_rl=pd_emp_rl,
    save=save_all)

print("%s: %f min" % (study_name, (time() - time0) / 60.0))