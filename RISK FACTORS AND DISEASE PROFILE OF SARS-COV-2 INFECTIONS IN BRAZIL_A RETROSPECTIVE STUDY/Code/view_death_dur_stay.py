#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file makes the Supplementary Figure 5, it needs the filter_SRAG.py
results to run. 
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

data_init = pd.read_csv('../Data/SRAG_filtered_morb.csv')
data_init = data_init[(data_init.EVOLUCAO==1)|(data_init.EVOLUCAO==2)]
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')
data_init['ti'] = (data_init.DT_EVOLUCA - data_init.DT_INTERNA).dt.days


cases, td = np.histogram(data_init.ti, bins=np.arange(0, 90))
deaths, td = np.histogram(data_init.ti[data_init.EVOLUCAO==2], bins=np.arange(0, 90))

td = td[:-1]

plt.figure()
plt.plot(td, deaths/cases)

plt.ylabel('Mortality')
plt.xlabel('Stay Duration (days)')
plt.xlim([-0.5,89])
plt.ylim([0.2,0.7])
plt.grid()
plt.tight_layout()
s = {'days': td, 'mortality': deaths/cases}
s = pd.DataFrame(s)
s.to_csv('../Results/mort_dur_hosp.csv', index=False)

ts = [4, 12, 40]

for t in ts:
    plt.plot([t,t], [0.2, 0.7], '--r')    
    
plt.savefig('../Figures/SFig5.png')
