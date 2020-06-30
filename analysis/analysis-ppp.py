# -*- coding: utf-8 -*-
"""
Simulation study: Poisson point process (ppp)
"""

from time import time

import numpy as np

import util, priors

#=============================================================================#

# Saves charts and LateX snippets at util.save_path
save_all = True

time0 = time()

study_name = "ppp"

# Simulating data #-----------------------------------------------------------#
#=============================================================================#

ppp_theta = [25.0, 5.0, 0.2]
ppp_M = 54
ppp_n_exceed = 86

ppp_data = util.poisson_point_process(
    ppp_theta,
    M=ppp_M,
    no_exceed=ppp_n_exceed,
    load=True,
    save=False,
    name=study_name)

ppp_data.draw(save=save_all)

# Constructing priors #-------------------------------------------------------#
#=============================================================================#

p = [0.1, 0.01, 0.001]

ppp_pi = priors.all_priors(
    p, 
    0.5,
    theta=ppp_theta,
    variance=5,
    name=study_name,
    save=save_all)
#%%
# MCMC sampling #-------------------------------------------------------------#
#=============================================================================#

# Set this to i to only sample from the i-th prior.
# Useful for tuning parameters.
isolate = None

for i in range(len(ppp_pi)):
    if isolate is not None and i is not isolate:
        continue
    
    ppp_pi[i].mcmc(
        "prior",
        [25.0, 0.2, 0.25],
        [
            [20.0, 0.5, 0.1],
            None,
            None,
            [40.0, 1, 0.3],
            [100.0, 2.5, 0.8],
            [20.0, 0.5, 0.1]][i],
        save=save_all)
    
    ppp_pi[i].mcmc(
        "post",
        [26.0, 1.8, 0.15],
        [
            [4.0, 0.5, 0.1],
            [4.0, 0.5, 0.2],
            [4.0, 0.5, 0.4],
            [4.0, 0.5, 0.3],
            [4.0, 0.5, 0.4],
            [4.0, 0.5, 0.1]][i],
        data=ppp_data,
        save=save_all)
#%%
# Visualisation #-------------------------------------------------------------#
#=============================================================================#

util.draw_list_priors_marginals(
    ppp_pi,
    support={
        "q": [[15.0, 75.0], [20.0, 140.0], [20.0, 250.0]],
        "theta": [[-50.0, 80.0], [-4.0, 5.0], [-0.6, 1]]},
    save=save_all)
#%%
# Return level plot #---------------------------------------------------------#
#=============================================================================#

# Analytic return level:

X = np.logspace(0.0001, 3.0, 20)
Y = [util.quantile(ppp_theta, 1.0 - 1.0 / x) for x in X]
ppp_true_rl = [X, Y]

# Empirical return level:

# Number of samples
n = 5000

ppp_emp_rl = util.poisson_point_process(
    ppp_theta,
    M=ppp_M * n,
    no_exceed=ppp_n_exceed * n).emp_quant

for i, prior in enumerate(ppp_pi):
    util.plot_return_level(
        prior,
        true_rl=ppp_true_rl,
        emp_rl=ppp_emp_rl,
        ylim=200,
        save=save_all)
    
print("%s: %f min" % (study_name, (time() - time0) / 60.0))