#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:49:50 2022

@author: lhunlindeion
"""
import numpy as np
from scipy.optimize import minimize, root
import datetime
import pandas as pd

ref = pd.to_datetime(datetime.date(2019, 12, 31))

data_init = pd.read_csv('../SRAG_filtered_morb.csv')
data_init['MORTE'] = (data_init.EVOLUCAO == 2)
states = np.r_[np.array([ 'BR' ]), data_init.SG_UF_INTE.unique()]
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')
# data_init['tth'] = (data_init.DT_INTERNA - data_init.DT_SIN_PRI).dt.days
# data_init['ttn'] = (data_init.DT_NOTIFIC - data_init.DT_SIN_PRI).dt.days
# data_init.loc[pd.isna(data_init.VACINA_COV),'VACINA_COV'] = 0

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
data_init['AGE_GRP'] = ''
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i]),'AGE_GRP'] = 'AG85+'
    else:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i])&(data_init.NU_IDADE_N<ages[i+1]), 'AGE_GRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])

ibpv = [data_init.ibp.quantile(x) for x in [0.0,0.2,0.4,0.6,0.8,1.0]]
names = [ 'BDI_' + i for i in ['0', '1', '2', '3', '4']]
data_init['BDI_GRP'] = ''
for i in range(5):
    if i == 4:
        data_init.loc[(data_init.ibp>=ibpv[i]),'BDI_GRP'] = names[i]
    else:
        data_init.loc[(data_init.ibp>=ibpv[i])&(data_init.ibp<ibpv[i+1]), 'BDI_GRP'] = names[i]

trad_raca = {1:'Branca', 2:'Preta', 3:'Amarela', 4:'Parda', 5:'Indigena'}
data_init['RACA'] = data_init['CS_RACA'].map(trad_raca)

ages = {loc:(data_init.AGE_GRP==loc).sum() for loc in data_init.AGE_GRP.unique()}

print(ages)
sexs = {loc:(data_init.CS_SEXO==loc).sum() for loc in data_init.CS_SEXO.unique()}
print(sexs)
raca = {loc:(data_init.RACA==loc).sum() for loc in data_init.RACA.unique()}
raca[np.nan] = pd.isna(data_init.RACA).sum()
print(raca)
bdi = {loc:(data_init.BDI_GRP==loc).sum() for loc in data_init.BDI_GRP.unique()}
print(bdi)

gr_risco = ['PNEUMOPATI', 'IMUNODEPRE', 'OBESIDADE', 'SIND_DOWN', \
            'RENAL', 'NEUROLOGIC', 'DIABETES', 'PUERPERA', 'OUT_MORBI', \
            'HEMATOLOGI', 'ASMA', 'HEPATICA', 'CARDIOPATI']
no_risco = np.ones((len(data_init)))
grupos = dict()
for risco in gr_risco:
    grupos[risco] = (data_init[risco]==1).sum()
    no_risco = no_risco * (1-(data_init[risco] == 1))

grupos['NENHUM'] = no_risco.sum()
print(grupos)