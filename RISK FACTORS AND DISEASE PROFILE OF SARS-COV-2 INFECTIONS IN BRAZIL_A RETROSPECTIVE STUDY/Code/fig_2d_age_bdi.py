#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine to generate Supplementary Figure 3. It generate separate pngs for
the individual plots in the figure. It needs the generate_sf3_data to be
run beforehand
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sah = pd.read_csv('../Results/mean_h.csv')
sau = pd.read_csv('../Results/mean_u.csv')
s = (10,8)


X = np.array([-1.7627 , -1.31348, -0.86426, -0.41504,  0.03418,  0.4834 ,
        0.93262,  1.38184,  1.83106,  2.28028,  2.7295 ])
Y = np.array([0, 18, 30, 40, 50, 65, 75, 85, 95])


Z = (sah.n_death/(sah.n_death+sah.n_cure)).to_numpy().reshape(s)
N = (sah.n_death+sah.n_cure).to_numpy().reshape(s)
Z[N<100] = np.nan
plt.figure()
plt.pcolormesh(X, Y, Z.T)
plt.colorbar()
CS = plt.contour(0.5*(X[1:]+X[:-1]), 0.5*(Y[1:]+Y[:-1]), Z.T, colors='w', levels=[0.1,0.2,0.4, 0.6, 0.75])#
plt.gca().clabel(CS, inline=True)
plt.xlabel('BDI')
plt.ylabel('Age')
plt.title('Mortality (Non-ICU patients)')
plt.tight_layout()
plt.savefig('../Figures/contour_HOSP_mortality.png')



Z = (sau.n_death/(sau.n_death+sau.n_cure)).to_numpy().reshape(s)
N = (sau.n_death+sau.n_cure).to_numpy().reshape(s)
Z[N<100] = np.nan
plt.figure()
plt.pcolormesh(X, Y, Z.T)
plt.colorbar()
CS = plt.contour(0.5*(X[1:]+X[:-1]), 0.5*(Y[1:]+Y[:-1]), Z.T, colors='w', levels=[0.1,0.2,0.4, 0.6, 0.75])#
plt.gca().clabel(CS, inline=True)
plt.xlabel('BDI')
plt.ylabel('Age')
plt.title('Mortality (ICU patients)')
plt.tight_layout()
plt.savefig('../Figures/contour_ICU_mortality.png')