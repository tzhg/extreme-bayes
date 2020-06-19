# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:45:42 2020

@author: tonyz
"""


par = [None for _ in range(3)]
par2 = [None for _ in range(3)]

for i in range(3):
    target = lambda x: gamma.pdf(x, para[i][0], scale=1.0 / para[i][1])
    
    def KL(par):
        mu, logsigma = par
        
        dist = lambda x: truncnorm.pdf(x, - mu / np.exp(logsigma), np.inf, loc=mu, scale=np.exp(logsigma))
        
        upper_bound = 100
        
        def integrand(x):
            pdf1 = dist(x)
            pdf2 = target(x)
            pdf1 = max(pdf1, 1e-140)
            pdf2 = max(pdf2, 1e-140)
            return pdf1 * np.log(pdf1 / pdf2)
        
        X = np.linspace(0.0, upper_bound, 50)
        return upper_bound * np.mean([integrand(x) for x in X])
    
    res = minimize(KL, (40.0, 1.0))
    
    mu, logsigma = res.x
    
    print([mu, np.exp(logsigma)])
    
    par[i] = [mu, np.exp(logsigma)]
    
    def KL2(loglamb):
        print(np.exp(loglamb))
        dist = lambda x: expon.pdf(x, scale=1.0 / np.exp(loglamb))
        
        upper_bound = 100
        
        def integrand(x):
            pdf1 = dist(x)
            pdf2 = target(x)
            pdf1 = max(pdf1, 1e-140)
            pdf2 = max(pdf2, 1e-140)
            return pdf1 * np.log(pdf1 / pdf2)
        
        X = np.linspace(0.0, upper_bound, 50)
        return upper_bound * np.mean([integrand(x) for x in X])
    
    res = minimize(KL2, 1.0)
    
    loglamb = res.x
    
    print(np.exp(lamb))
    
    par2[i] = np.exp(lamb)
    
    
    
X = np.linspace(0, 100, 100)
fig, ax = plt.subplots(nrows=1, ncols=3)
for j in range(3):
    dist = lambda x: truncnorm.pdf(x, - par[j][0] / par[j][1], np.inf, loc=par[j][0], scale=par[j][1])
    dist2 = lambda x: expon.pdf(x, scale=1.0 / par2[j])
    target = lambda x: gamma.pdf(x, para[j][0], scale=1.0 / para[j][1])
    cell = ax[j]
    cell.plot(X, [target(x) for x in X])
    cell.plot(X, [dist(x) for x in X], "r")
    cell.plot(X, [dist2(x) for x in X], "g")
fig.set_size_inches(9, 3)
fig.tight_layout()
plt.show()
