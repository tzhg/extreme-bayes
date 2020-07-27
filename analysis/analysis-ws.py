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

theta = data.fit_GEV(save=True)

data = data.optimal_M(theta[2])

# There are 212 days from November to May
data.set_obs_in_year(212)

# Constructing priors #-------------------------------------------------------#
#=============================================================================#

p = np.array([0.1, 0.01, 0.001])
    
pi = priors.all_priors(
    p,
    [26.45, 33.636, 48],
    var=27,
    name=study_name)

# MCMC sampling #-------------------------------------------------------------#
#=============================================================================#

for i in range(4):
    if pi[i].prior["proper"]:
        pi[i].get_samples(
            "prior",
            [
                [25.0, 2, 0], # 3P I
                None,
                [25.0, 2, 0], # 3P ME
                None][i],
            [   
                [25.0, 2, 0.6], # 3P I
                None,
                [25.0, 2, 0.8], # 3P ME
                None][i],
            p,
            save=save_all)
    
    if pi[i].post["proper"]:
        pi[i].get_samples(
            "post",
            [
                [22.5, 1.0, 0.2], # 3P I
                [22.5, 1.0, 0.3], # 2P I
                [22.5, 1.0, 0.1], # 3P ME
                [22.5, 1.0, 0.4]][i], # 2P ME
            [
                [1.0, 0.45, 0.25], # 3P I
                [1.0, 0.5, 0.4], # 2P I
                [1.0, 0.4, 0.3], # 3P ME
                [1.0, 0.5, 0.4]][i], # 2P ME
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