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

theta = [25.0, 5.0, 0.2]

# Same as in Coles et al.
n = 19710
no_exceed = 86

data = util.poisson_point_process(
    theta,
    n=n,
    no_exceed=no_exceed,
    load=True,
    save=False,
    name=study_name)

data.draw(save=save_all)

data = data.optimal_M(theta[2])

# Plots fit of annual maxima
data.set_obs_in_year(365)
theta_annual = data.theta_annual(theta)
data.fit_GEV(theta=theta_annual, save=True)

# Constructing priors #-------------------------------------------------------#
#=============================================================================#

p = np.array([0.1, 0.01, 0.001])

pi = priors.all_priors(
    p,
    util.quantile(theta, 1.0 - p),
    27,
    name=study_name)

# MCMC sampling #-------------------------------------------------------------#
#=============================================================================#

for i in range(4):
    if pi[i].prior["proper"]:
        pi[i].get_samples(
            "prior",
            [
                [25.0, 2.0, 0.2], # 3P I
                None,
                [25.0, 2.0, 0.0], # 3P ME
                None][i],
            [   
                [25.0, 1.0, 0.2], # 3P I
                None,
                [25.0, 0.5, 0.1], # 3P ME
                None][i],
            p,
            save=save_all)
    
    if pi[i].post["proper"]:
        pi[i].get_samples(
            "post",
            [
                [25.0, 2.0, 0.2], # 3P I
                [25.0, 2.0, 0.2], # 2P I
                [25.0, 2.0, 0.2], # 3P ME
                [25.0, 2.0, 0.2]][i], # 2P ME
            [
                [3.0, 0.45, 0.25], # 3P I
                [3.0, 0.5, 0.4], # 2P I
                [3.0, 0.4, 0.15], # 3P ME
                [3.0, 0.5, 0.4]][i], # 2P ME
            p,
            data=data,
            save=save_all)
        
# Plotting marginals #--------------------------------------------------------#
#=============================================================================#

for prior in pi:
    prior.set_marginals()

util.draw_list_priors_marginals(
    pi,
    support={
        "theta": {
            "prior": [[0.0, 50.0], [-3, 25], [-0.5, 1.0]],
            "post": [[22.5, 28], [3.5, 11.5], [-0.1, 0.75]]},
        "q": {
            "prior": [[20.0, 55.0], [35.0, 120.0], [60.0, 275.0]],
            "post": [[35, 55.0], [35.0, 120.0], [60.0, 275.0]]}},
    save=save_all)

# Return level plot #---------------------------------------------------------#
#=============================================================================#

# Asymptotic return level
    
X = np.logspace(0.0001, 4.0, 20)
Y = [util.quantile(theta_annual, 1.0 - 1.0 / x) for x in X]
analytic_rl = [X, Y]

# Return level plot

util.plot_return_level(
    pi,
    analytic_rl=analytic_rl,
    simul_rl=None,
    save=save_all)
    
print("%s: %f min" % (study_name, (time() - time0) / 60.0))