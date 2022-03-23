#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the time series for cases and deaths separated by BDI classes.

"""

import numpy as np
# from scipy.optimize import minimize, root
import datetime
import pandas as pd
import matplotlib.pyplot as plt


ref = pd.to_datetime(datetime.date(2019, 12, 31))
last = pd.to_datetime(datetime.date(2021, 8, 15))
CFR = 0.006

# t_end = 60
t0 = 46
data_init = pd.read_csv('../Data/SRAG_filtered_morb.csv')
data_init['MORTE'] = (data_init.EVOLUCAO == 2)
states = np.r_[np.array([ 'BR' ]), data_init.SG_UF_INTE.unique()]
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')

ibpv = [data_init.ibp.quantile(x) for x in [0.0,0.2,0.4,0.6,0.8,1.0]]
srag_raw = data_init[(data_init.EVOLUCAO==2) & (data_init.DT_SIN_PRI<=last)]
srag_c = data_init[(data_init.DT_SIN_PRI<=last)]
data_init['tnd'] = (data_init.DT_EVOLUCA - data_init.DT_SIN_PRI).dt.days
n_d, t_d = np.histogram(data_init.tnd, bins=np.arange(0,160))
t_d = t_d[:-1]


#%%
sg_raw = pd.read_csv('../Data/sg.csv')


ibp = pd.read_csv('../Data/pop_ibp.csv')
sg_raw['ibp'] = np.nan
for x, valor in zip(ibp.Cod, ibp.ip_vl_n):
    sg_raw.loc[sg_raw.municipioIBGE == x,'ibp'] = valor

uc = ['dataInicioSintomas', 'ibp', 'evolucaoCaso']
sg_raw = sg_raw[uc]

for col in sg_raw.columns:
    if col[:4] == 'data':
        sg_raw.loc[:,col] = pd.to_datetime(sg_raw[col], format='%Y-%m-%d', errors='coerce')

sg_raw = sg_raw[(~pd.isna(sg_raw.dataInicioSintomas)) & (sg_raw.dataInicioSintomas <= last)]

sg_to_remove = [6, 3, 2]
for toremove in sg_to_remove:
    print(toremove, (sg_raw.evolucaoCaso==toremove).sum())
    sg_raw = sg_raw[sg_raw.evolucaoCaso != toremove]

#%%


day_sg = (sg_raw.dataInicioSintomas - ref).dt.days
day_srag = (srag_raw.DT_SIN_PRI - ref).dt.days
day_sragc = (srag_c.DT_SIN_PRI - ref).dt.days
cases_srag, td = np.histogram(day_sragc, bins=np.arange(t0, (last-ref).days))
cases_sg, t_d = np.histogram(day_sg, bins=np.arange(t0, (last-ref).days))
fatalities, t_d = np.histogram(day_srag, bins=np.arange(t0, (last-ref).days))
t_d = t_d[:-1]
cases = cases_sg + cases_srag
t_d = t_d[cases>0]
fatalities = fatalities[cases>0]
cases = cases[cases>0]

s = {'t': t_d, 'deaths': fatalities, 'cases': cases,
     'cases_cum': np.cumsum(cases), 'nCFR': fatalities/cases, 
     'cases_adj': fatalities/CFR ,'cases_adj_cum': np.cumsum(fatalities/CFR)}
s = pd.DataFrame(s)
s['date'] = ref + pd.to_timedelta(t_d, unit='day')
s.to_csv(f'../Results/cfrs_all.csv', index=False)
nsep = 5
saida = []
for j in range(nsep):
    print(j)
    if j == nsep-1:
        srag = srag_raw[srag_raw.ibp>=ibpv[j]]
        sg = sg_raw[sg_raw.ibp>=ibpv[j]]
        src = srag_c[srag_c.ibp>=ibpv[j]]
    else:
        srag = srag_raw[(srag_raw.ibp>=ibpv[j])&(srag_raw.ibp<ibpv[j+1])]
        sg = sg_raw[(sg_raw.ibp>=ibpv[j])&(sg_raw.ibp<ibpv[j+1])]
        src = srag_c[(srag_c.ibp>=ibpv[j])&(srag_c.ibp<ibpv[j+1])]
    
    day_sg = (sg.dataInicioSintomas - ref).dt.days
    day_srag = (srag.DT_SIN_PRI - ref).dt.days
    day_sragc = (src.DT_SIN_PRI - ref).dt.days
    cases_srag, td = np.histogram(day_sragc, bins=np.arange(t0, (last-ref).days))
    cases_sg, t_d = np.histogram(day_sg, bins=np.arange(t0, (last-ref).days))
    fatalities, t_d = np.histogram(day_srag, bins=np.arange(t0, (last-ref).days))
    t_d = t_d[:-1]
    cases = cases_sg + cases_srag
    t_d = t_d[cases>0]
    fatalities = fatalities[cases>0]
    cases = cases[cases>0]
    saida.append([t_d, cases, fatalities])
#%%


for i, temp in enumerate(saida):
    t, case, fatal = temp
    s = {'t': t, 'deaths': fatal, 'cases': case, 'cases_cum': np.cumsum(case), 
         'nCFR': fatal/case, 'case_adj': fatal/CFR,
         'cases_adj_cum': np.cumsum(fatal/CFR)}
    s = pd.DataFrame(s)
    s['date'] = ref + pd.to_timedelta(t, unit='day')
    s.to_csv(f'../Results/Class_{i}_series.csv', index=False)
    