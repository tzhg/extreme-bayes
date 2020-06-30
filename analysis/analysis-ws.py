# -*- coding: utf-8 -*-
"""
Real data: wind speed (ws)
"""

from time import time

import util, priors

#=============================================================================#

# Saves charts and LateX snippets at util.save_path
save_all = True

time0 = time()

study_name = "ws"

# Obtaining data #------------------------------------------------------------#
#=============================================================================#

ws_data_raw = util.load_data(study_name)

ws_M = len(ws_data_raw) / 365.0
ws_u = 25.0

ws_data = util.GEVData(ws_u, ws_data_raw, ws_M, study_name)

ws_data.draw(save=save_all)

# Constructing priors #-------------------------------------------------------#
#=============================================================================#

p = [0.1, 0.01, 0.001]

ws_theta = ws_data.fit_GEV(True)
    
ws_pi = priors.all_priors(
    p,
    0.5,
    theta=ws_theta,
    variance=5,
    name=study_name,
    save=save_all)
#%%
# MCMC sampling #-------------------------------------------------------------#
#=============================================================================#

# Set this to i to only sample from the i-th prior.
# Useful for tuning parameters.
isolate = None

for i in range(len(ws_pi)):
    if isolate is not None and i is not isolate:
        continue
    
    ws_pi[i].mcmc(
        "prior",
        [27.0, 3.7, -0.12],
        [
            [25.0, 3, 1],
            None,
            None,
            [30.0, 5, 1],
            [100.0, 5, 1.5],
            [20.0, 2, 0.8]][i],
        save=save_all)
    
    ws_pi[i].mcmc(
        "post",
        [27.0, 0.8, 0.15],
        [
            [2.0, 0.5, 0.5],
            [2.0, 0.5, 0.6],
            [2.0, 0.5, 0.6],
            [2.0, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [2.0, 0.5, 0.5]][i],
        data=ws_data,
        save=save_all)

#%%
# Visualisation #-------------------------------------------------------------#
#=============================================================================#

util.draw_list_priors_marginals(
    ws_pi,
    support={
        "q": [[0.0, 50.0], [0.0, 110.0], [0.0, 300.0]],
        "theta": [[-10.0, 50.0], [-2.0, 4.0], [-0.5, 1]]},
    save=save_all)

# Return level plot #---------------------------------------------------------#
#=============================================================================#

for i, prior in enumerate(ws_pi):
    util.plot_return_level(
        prior,
        ylim=250,
        save=save_all)

print("%s: %f min" % (study_name, (time() - time0) / 60.0))