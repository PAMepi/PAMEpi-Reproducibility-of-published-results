#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates the data for Fig 2C-D.
The sex plot data are saved in the files mort_sexo_ICU.csv and
mort_sexo_HOSP.csv, while the vaccination data is saved in the files
mort_vacina_ICU.csv and mort_vacina_HOSP.csv.

Column descriptions (X=male, female, vac,  unvac):
    age_grp = age group
    mean_age_X = mean value of the age inside this age group with the condition X
    mort_X = mortality in this age group with the condition X
"""

import numpy as np
# from scipy.optimize import minimize, root
import datetime
import pandas as pd
import matplotlib.pyplot as plt


ref = datetime.date(2019, 12, 31)

data_init = pd.read_csv('../Data/SRAG_filtered_morb.csv')

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
data_init['AGE_GRP'] = ''
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i]),'AGE_GRP'] = 'AG85+'
    else:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i])&(data_init.NU_IDADE_N<ages[i+1]), 'AGE_GRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])
data_init = data_init[data_init.AGE_GRP != '']

data_init['VACC'] = (data_init.VACINA_COV==1)

names = ['all', 'ICU', 'HOSP']
datas = [data_init, data_init[~pd.isna(data_init.DT_ENTUTI)], \
         data_init[pd.isna(data_init.DT_ENTUTI)]]

for name, data in zip(names, datas):
    vac = {'age_grp':[], 'mean_age_vac':[], 'mean_age_unvac':[], 'mort_vac':[], 'mort_unvac':[]}
    sex = {'age_grp':[], 'mean_age_male':[], 'mean_age_female':[], 'mort_male':[], 'mort_female':[]}
    
    ags = data.AGE_GRP.unique()
    ags.sort()
    for ag in ags:
        vac['age_grp'].append(ag)
        sex['age_grp'].append(ag)
        vac['mean_age_vac'].append(data.NU_IDADE_N[(data.AGE_GRP==ag)&(data.VACC==True)].mean())
        vac['mean_age_unvac'].append(data.NU_IDADE_N[(data.AGE_GRP==ag)&(data.VACC==False)].mean())
    
        sex['mean_age_male'].append(data.NU_IDADE_N[(data.AGE_GRP==ag)&(data.CS_SEXO=='M')].mean())
        sex['mean_age_female'].append(data.NU_IDADE_N[(data.AGE_GRP==ag)&(data.CS_SEXO=='F')].mean())
        
        mort = data[data.EVOLUCAO == 2]
        tot = data[(data.EVOLUCAO == 1) | (data.EVOLUCAO == 2)]
        print(name, ag, len(mort[(mort.AGE_GRP==ag)&(mort.CS_SEXO=='M')]), len(mort[(mort.AGE_GRP==ag)&(mort.CS_SEXO=='F')]) )
        vac['mort_vac'].append(len(mort[(mort.AGE_GRP==ag)&(mort.VACC==True)])\
                               / len(tot[(tot.AGE_GRP==ag)&(tot.VACC==True)]))
        vac['mort_unvac'].append(len(mort[(mort.AGE_GRP==ag)&(mort.VACC==False)])\
                               / len(tot[(tot.AGE_GRP==ag)&(tot.VACC==False)]))
        sex['mort_male'].append(len(mort[(mort.AGE_GRP==ag)&(mort.CS_SEXO=='M')])\
                               / len(tot[(tot.AGE_GRP==ag)&(tot.CS_SEXO=='M')]))
        sex['mort_female'].append(len(mort[(mort.AGE_GRP==ag)&(mort.CS_SEXO=='F')])\
                               / len(tot[(tot.AGE_GRP==ag)&(tot.CS_SEXO=='F')]))
    
    vac = pd.DataFrame(vac)
    sex = pd.DataFrame(sex)
    vac.to_csv(f'../Results/mort_vacina_{name}.csv', index=False)
    sex.to_csv(f'../Results/mort_sexo_{name}.csv', index=False)