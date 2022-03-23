#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the data for Fig 2A. For the figures it is enough to plot ibp_mean 
(mean BDI value in the BDI window) versus the mortality for each output file.
"""

import pandas as pd
import numpy as np


data0 = pd.read_csv('../Data/SRAG_filtered_morb.csv')
data0 = data0[(~pd.isna(data0.ibp))]

saida_H = {'n_death':[], 'n_cure':[], 'mortality':[], 's_mort':[],\
           'ibp_min':[], 'ibp_mean':[], 'ibp_max':[]}
saida_U = { 'n_death':[], 'n_cure':[], 'mortality':[],\
           'ibp_min':[], 'ibp_mean':[], 'ibp_max':[], 's_mort':[],}

nsep = 10
ibps = np.linspace(data0.ibp.min(), data0.ibp.max(), nsep+1)

for i in range(nsep):
    print(i)
    if i == nsep-1:
        data = data0[(data0.ibp>=ibps[i])] 
    else:
        data = data0[(data0.ibp>=ibps[i])&(data0.ibp<ibps[i+1])]
    
    ibpm = data.ibp.mean()
    ibpi = [data.ibp.min(), data.ibp.max()]
    saida_H['ibp_mean'].append(ibpm)
    saida_U['ibp_mean'].append(ibpm)
    saida_H['ibp_max'].append(ibpi[1])
    saida_U['ibp_max'].append(ibpi[1])
    saida_H['ibp_min'].append(ibpi[0])
    saida_U['ibp_min'].append(ibpi[0])
    
    dICU = np.sum((~pd.isna(data.DT_ENTUTI))&(data.EVOLUCAO==2))
    cICU = np.sum((~pd.isna(data.DT_ENTUTI))&(data.EVOLUCAO==1))
    pICU = dICU / (dICU + cICU)
    spICU = np.sqrt(pICU * (1 - pICU) / (dICU + cICU)) 
    dH = np.sum((pd.isna(data.DT_ENTUTI))&(data.EVOLUCAO==2))
    cH = np.sum((pd.isna(data.DT_ENTUTI))&(data.EVOLUCAO==1))
    pH = dH / (dH + cH)
    spH = np.sqrt(pH * (1 - pH) / (dH + cH)) 
    saida_H['n_death'].append(dH)
    saida_U['n_death'].append(dICU)
    saida_H['n_cure'].append(cH)
    saida_U['n_cure'].append(cICU)
    saida_H['mortality'].append(pH)
    saida_U['mortality'].append(pICU)
    saida_H['s_mort'].append(spH)
    saida_U['s_mort'].append(spICU)
    
sah = pd.DataFrame(saida_H)
sah.to_csv('../Results/bdi_means_h.csv', index=False)
sah = pd.DataFrame(saida_U)
sah.to_csv('../Results/bdi_means_u.csv', index=False)