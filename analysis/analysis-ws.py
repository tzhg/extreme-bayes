# -*- coding: utf-8 -*-
"""
Real data: wind speed (ws)
"""

from time import time

import numpy as np

import util, priors

#=============================================================================#

# Saves charts and LateX snippets at util.save_path
save_all = True

time0 = time()

study_name = "ws"

# Obtaining data #------------------------------------------------------------#
#=============================================================================#

data_raw = util.load_data(study_name)

u = 21.0

data = util.GEVData(u, data_raw, name=study_name)

data.draw(save=save_all)

# Fits GEV model
theta = data.fit_GEV(save=True)

# Chooses optimal value of M
data = data.optimal_M(theta[2])

# There are 212 days from November to May
data.set_obs_in_year(212)

# Constructing priors #-------------------------------------------------------#
#=============================================================================#

p = np.array([0.1, 0.01, 0.001])
    
pi = priors.all_priors(
    p,
    qu=[26.45, 33.636, 48],
    var=[27] * 3,
    name=study_name)

# MCMC sampling #-------------------------------------------------------------#
#=============================================================================#

for i in range(4):
    if pi[i].prior["proper"]:
        pi[i].get_samples(
            "prior",
            [
                [25.0, 2.0, 0.0], # k = 3, I copula
                None,
                [25.0, 2.0, 0.0], # k = 3, ME copula
                None][i],
            [   
                [30.0, 2.0, 0.6], # k = 3, I copula
                None,
                [25.0, 2.0, 0.8], # k = 3, ME copula
                None][i],
            p,
            save=save_all)
    
    if pi[i].post["proper"]:
        pi[i].get_samples(
            "post",
            [
                [21.0, 0.75, 0.2],     # k = 3, I copula
                [21.0, 0.75, 0.3],     # k = 2, I copula
                [21.0, 0.75, 0.3],     # k = 3, ME copula
                [21.0, 0.75, 0.4]][i], # k = 2, ME copula
            [
                [1.0, 0.45, 0.25],   # k = 3, I copula
                [1.0, 0.5, 0.4],     # k = 2, I copula
                [1.0, 0.5, 0.3],     # k = 3, ME copula
                [1.0, 0.5, 0.4]][i], # k = 2, ME copula
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
            "prior": [[10.0, 35.0], [-4, 8.0], [-1, 2]],
            "post": [[20.5, 24], [-1, 6], [-0.15, 1]]},
        "q": {
            "prior": [[10.0, 47.50], [10.0, 125.0], [10.0, 125.0]],
            "post": [[25.0, 47.5], [20.0, 130.0], [0.0, 375.0]]}},
    save=save_all)

# Return level plot #---------------------------------------------------------#
#=============================================================================#

util.plot_return_level(
    pi,
    save=save_all)

print("%s: %f min" % (study_name, (time() - time0) / 60.0))