#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counts the number of patients with each feature for the group with ethnic/racial
background missing and filled 
"""

import numpy as np
import pandas as pd

data_init = pd.read_csv('../Data/SRAG_filtered_morb.csv')
data_init['MORTE'] = (data_init.EVOLUCAO == 2)
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')

data_init['ti'] = (data_init.DT_EVOLUCA - data_init.DT_INTERNA).dt.days

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
data_init['AGE_GRP'] = 'NA'
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i]),'AGE_GRP'] = 'AG85+'
    else:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i])&(data_init.NU_IDADE_N<ages[i+1]), 'AGE_GRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])

ibpv = [data_init.ibp.quantile(x) for x in [0.0,0.2,0.4,0.6,0.8,1.0]]
names = [ 'BDI_' + i for i in ['0', '1', '2', '3', '4']]
data_init['BDI_GRP'] = 'NA'
for i in range(5):
    if i == 4:
        data_init.loc[(data_init.ibp>=ibpv[i]),'BDI_GRP'] = names[i]
    else:
        data_init.loc[(data_init.ibp>=ibpv[i])&(data_init.ibp<ibpv[i+1]), 'BDI_GRP'] = names[i]

tis = [0, 4, 12, 40, np.inf]
nsep = len(tis) - 1
data_init['TINT'] = 'NA'
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.ti>=tis[i]),'TINT'] = 'TM40'
    else:
        data_init.loc[(data_init.ti>=tis[i])&(data_init.ti<tis[i+1]), 'TINT'] = 'T{}t{}'.format(tis[i],tis[i+1])

data_init['eUTI'] = ~pd.isna(data_init.DT_ENTUTI)
data_init['NVACC'] = (data_init.VACINA_COV!=1)

gr_risco = ['PNEUMOPATI', 'IMUNODEPRE', 'OBESIDADE', 'SIND_DOWN', \
            'RENAL', 'NEUROLOGIC', 'DIABETES', 'PUERPERA', 'OUT_MORBI', \
            'HEMATOLOGI', 'ASMA', 'HEPATICA', 'CARDIOPATI']

data_init['NO_COMORB'] = 1
for col in gr_risco:
    data_init[col] = (data_init[col] == 1)
    data_init.loc[data_init[col],'NO_COMORB'] = 0
    
trad_raca = {1:'Branca', 2:'Preta', 3:'Amarela', 4:'Parda', 5:'Indigena', 9:'Desconhecida'}

data_init['RACA'] = data_init['CS_RACA'].map(trad_raca)
data_init.loc[pd.isna(data_init.RACA), 'RACA'] = 'Desconhecida'
data_init.loc[pd.isna(data_init.CS_SEXO),'CS_SEXO'] = 'I'

srag_raca = data_init[data_init.RACA != 'Desconhecida']
srag_sem = data_init[data_init.RACA == 'Desconhecida']
print(len(srag_raca), len(srag_sem))
Nr = len(srag_raca)
Ns = len(srag_sem)
output = {'variable':['TOTAL'], 'count_com_raca':[len(srag_raca)],\
          'count_sem_raca':[len(srag_sem)], 'frac_com_raca':[1.],\
          'frac_sem_raca':[1.]}

binarios = ['MORTE', 'eUTI', 'NVACC'] + gr_risco + ['NO_COMORB']
categoricos = ['BDI_GRP', 'AGE_GRP', 'TINT', 'CS_SEXO', "RACA"]
for row in binarios:
    output['variable'].append(row)
    output['count_com_raca'].append(srag_raca[row].sum())
    output['frac_com_raca'].append(srag_raca[row].sum()/Nr)
    output['count_sem_raca'].append(srag_sem[row].sum())
    output['frac_sem_raca'].append(srag_sem[row].sum()/Ns)

for row in categoricos:
    categs = np.sort(srag_raca[row].unique())
    if row == 'RACA':
        categs = np.r_[categs, ['Desconhecida']]
    for cats in categs:
        output['variable'].append(f"{row}_{cats}")
        output['count_com_raca'].append((srag_raca[row]==cats).sum())
        output['frac_com_raca'].append((srag_raca[row]==cats).sum()/Nr)
        output['count_sem_raca'].append((srag_sem[row]==cats).sum())
        output['frac_sem_raca'].append((srag_sem[row]==cats).sum()/Ns)

output = pd.DataFrame(output)
output.to_csv('../Results/table_race_vars.csv', index=False)