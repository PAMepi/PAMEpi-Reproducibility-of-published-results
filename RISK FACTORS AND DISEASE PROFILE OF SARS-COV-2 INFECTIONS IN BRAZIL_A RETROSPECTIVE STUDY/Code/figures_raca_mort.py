#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates the data for Figure 2B and saves in fig2_ethnicity, each row
depicts a age group (ag_grp column)
the file has two columns for each ethnicity:
Im_{} is the ICU mortality, and Hm_{} is the non-ICU mortality
"""
import numpy as np
# from scipy.optimize import minimize, root
import datetime
import pandas as pd

ref = datetime.date(2019, 12, 31)

data_init = pd.read_csv('../Data/SRAG_filtered_morb.csv')
data_init['MORTE'] = (data_init.EVOLUCAO == 2)
states = np.r_[np.array([ 'BR' ]), data_init.SG_UF_INTE.unique()]
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1

trans = {1:"White", 2:"Black", 3:"Yellow", 4:"Mixed", 5:'Indigenous', 9:"Unknown" }
out = dict()
out['ag_grp'] = ['AG{}t{}'.format(ages[i], ages[i+1]) for i in range(nsep-1)]
out['ag_grp'].append('AG85+')
for X in data_init['CS_RACA'].unique():
    if not np.isnan(X):
        nX = trans[X]
        out['Hm_{}'.format(nX)] = []
        out['Im_{}'.format(nX)] = []
        data0 = data_init[data_init.CS_RACA==X]
        dataU = data0[~pd.isna(data0.DT_ENTUTI)]
        nU = (((dataU.MORTE==1)) | (dataU.EVOLUCAO==1)).sum()
        pUd = ((dataU.MORTE == 1)).sum() / nU
        dataH = data0[pd.isna(data0.DT_ENTUTI)]
        nH = (((dataH.MORTE==1)) | (dataH.EVOLUCAO==1)).sum()
        pHd = ((dataH.MORTE == 1)).sum() / nH
        for i in range(nsep):
            if i == nsep-1:
                data = data0[(data0.NU_IDADE_N>=ages[i])]
            else:
                data = data0[(data0.NU_IDADE_N>=ages[i])&(data0.NU_IDADE_N<ages[i+1])]
            
            dataU = data[~pd.isna(data.DT_ENTUTI)]
            nU = (((dataU.MORTE==1)) | (dataU.EVOLUCAO==1)).sum()
            pUd = ((dataU.MORTE == 1)).sum() / nU
            out['Im_{}'.format(nX)].append(pUd)
            
            dataH = data[pd.isna(data.DT_ENTUTI)]
            nH = (((dataH.MORTE==1)) | (dataH.EVOLUCAO==1)).sum()
            pHd = ((dataH.MORTE == 1)).sum() / nH
            out['Hm_{}'.format(nX)].append(pHd)

out = pd.DataFrame(out)
out.to_csv('../Results/fig2_ethnicity.csv',  index=False)