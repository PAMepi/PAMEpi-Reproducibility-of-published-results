#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:11:56 2022

@author: lhunlindeion
"""

import numpy as np
import datetime
import pandas as pd


data_init = pd.read_csv('SRAG_filtered_morb.csv')
ibpv = [data_init.ibp.quantile(x) for x in [0.0,0.2,0.4,0.6,0.8,1.0]]

for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')


# data_init.CS_SEXO
ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
data_init['AGE_GRP'] =  ''
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i]),'AGE_GRP'] = 'AG85+'
    else:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i])&(data_init.NU_IDADE_N<ages[i+1]), 'AGE_GRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])

names = [ 'BDI_' + i for i in ['0', '1', '2', '3', '4']]
data_init['BDI_GRP'] = ''
for i in range(5):
    if i == 4:
        data_init.loc[(data_init.ibp>=ibpv[i]),'BDI_GRP'] = names[i]
    else:
        data_init.loc[(data_init.ibp>=ibpv[i])&(data_init.ibp<ibpv[i+1]), 'BDI_GRP'] = names[i]

trad_raca = {1:'Branca', 2:'Preta', 3:'Amarela', 4:'Parda', 5:'Indigena'}
data_init['RACA'] = data_init['CS_RACA'].map(trad_raca)
data_init['tv1ti'] = (data_init.DT_INTERNA - data_init.DOSE_1_COV).dt.days
data_init['tv2ti'] = (data_init.DT_INTERNA - data_init.DOSE_2_COV).dt.days

data = data_init[data_init.VACINA_COV==1]


# 0 = astrazeneca, 1 = coronavac, 2 = pfizer, 3 = janssen

#%%
data['LVAC'] = np.nan

def is_name_in_list(name, lnames):
    for rname in lnames:
        if name.find(rname) >= 0:
            return True
    return False


coronavac_list = ['CORONAVAC', 'BUTANTAN',  'SINOVAC', 'IB', 'SINO', 'BUTA', \
                  'CORAVAC', 'CORONA', 'CORONO', 'BUATN', 'CORANA', 'BUNTAN',\
                  'BUTU', 'SIONOVAC', 'CORONSVAC', 'COROVACAC', 'COORNAVAC',\
                  'COVONAVA', 'CORFONAVAC', 'TANTAN', 'SIVOVAC', 'CORO', 'CONO',\
                  'BHUTANTAN', 'CORANOVAC', 'CORNAVAC', 'CINOFARMA', 'BT',\
                  'BUTNTAN', 'SIVONAC', 'I.B.', 'CRONAVAC', 'SINAVAC', \
                  'SINVAC', 'CORNONAVAC', 'BUT', ]
coronavac_names = []
astrazeneca_list = ['ZENECA', 'OXFOR', 'ZENICA', 'FIOCRUZ', 'CRUZ', 'AZT', \
                    'CHADOX1', "INSTITUTO SERUM", 'COVISHILD', 'COVISHIELD', \
                    'ASTR', 'ATZ', 'FORD', 'OXF', '0XF', 'COVISHEID',
                    'COVISCHELD', 'ATRAZENCA', 'FIOCROZ', 'OSWALDO', 'SHIELD',\
                    'INDIA', 'FIOCFRUZ', 'CRIZ', 'ABX0529', 'OXOFRD', 'FIO RUZ',\
                    'COVIDCHIELD', 'CORVISHELD', 'COREISHIEL', 'ATRAZANICA', \
                    'FIOCURZ', 'INSTITUTO SERIUM', 'AZ', 'BIOMANGUINHO', \
                    'FIO CRUOZ', 'CHADOX', 'COVISHIED', 'SERUM', 'FIOCRUS' ]
astrazeneca_names = []
pfizer_list = ['FIZER', 'PFZES', 'PFIZAR', 'PIZER', 'IZER', 'BIONTECH', \
               'PZIFER', 'FAZER', 'EZER', 'PFI', 'PFZ', 'PFA', 'FAYZER', \
               'PZF', 'PZI',  'BIOTECNO', 'BNT162B2', 'PAIFFER', 'BIOTECH',\
               'COMIRNATY', 'PFYZER', 'BIO N TECH', 'PZHIER', 'FAISER'  ]
pfizer_names = []
janssen_list = ['JANSSEN', 'JANSEN', 'UNICA', 'JAH', 'JANSE', 'JENS', 'JANHSEN',
                'JASSEN', 'JONHSON', 'JONSSON', 'JANSON', 'JHONSON', 'JOHNSON',\
                'JOHSON', 'JHONNSONN', 'JONHOSON', 'JHANSSEN', 'JHANSEN',\
                'JONSSEN', 'JHONSOM', 'JASEN', 'JANSSER',  'JHONSSEN', \
                'JANSSEM', 'JONSHON', 'JANNSEN',    ]
janssen_names = []
others = []
for name in data.LAB_PR_COV.unique():
    if type(name) is str:
        if is_name_in_list(name, coronavac_list):
            coronavac_names.append(name)
            data.loc[data.LAB_PR_COV==name, 'LVAC'] = 1 
        elif is_name_in_list(name, astrazeneca_list):
            astrazeneca_names.append(name)
            data.loc[data.LAB_PR_COV==name, 'LVAC'] = 0 
        elif is_name_in_list(name, pfizer_list):
            pfizer_names.append(name)
            data.loc[data.LAB_PR_COV==name, 'LVAC'] = 2 
        elif is_name_in_list(name, janssen_list):
            janssen_names.append(name)
            data.loc[data.LAB_PR_COV==name, 'LVAC'] = 3 
        else:
            others.append([name, (data.LAB_PR_COV==name).sum()])
    else:
        others.append([name, pd.isna(data.LAB_PR_COV).sum()])


#%%
data_all = [data, data[data.EVOLUCAO==1], data[data.EVOLUCAO==2]]
name = ['_A', '_C', '_D']

sex = {}
for val in data.CS_SEXO.unique():
    for i in range(3):
        if val is np.nan:
            sex['NaN'+name[i]] = pd.isna(data_all[i].CS_SEXO).sum()
        else:
            sex[str(val)+name[i]] = (data_all[i].CS_SEXO==val).sum()
print('sex', sex)
age = {}
for val in data.AGE_GRP.unique():
    for i in range(3):
        if val is np.nan:
            age['NaN'+name[i]] = pd.isna(data_all[i].AGE_GRP).sum()
        else:
            age[str(val)+name[i]] = (data_all[i].AGE_GRP==val).sum()
print('age', age)
raca = {}
for val in data.RACA.unique():
    for i in range(3):
        if val is np.nan:
            raca['NaN'+name[i]] = pd.isna(data_all[i].RACA).sum()
        else:
            raca[str(val)+name[i]] = (data_all[i].RACA==val).sum()
print('raca', raca)
bdi = {}
for val in data.BDI_GRP.unique():
    for i in range(3):
        if val is np.nan:
            bdi['NaN'+name[i]] = pd.isna(data_all[i].BDI_GRP).sum()
        else:
            bdi[str(val)+name[i]] = (data_all[i].BDI_GRP==val).sum()
print('bdi',bdi)
labs = ['AZ', 'SC', 'PF', 'JS']
lab = {}
for val in data.LVAC.unique():
    for i in range(3):
        if np.isnan(val):
            lab['NaN'+name[i]] = pd.isna(data_all[i].LVAC).sum()
        else:
            lab[labs[int(val)]+name[i]] = (data_all[i].LVAC==val).sum()
print('lab',lab)

cycle = {}
for i in range(3):
    cycle['CO'+name[i]] = (pd.isna(data_all[i].DOSE_2_COV)|(data_all[i].LVAC==3)).sum()
    cycle['IN'+name[i]] = (len(data_all[i]) - cycle['CO'+name[i]])
print('cycle', cycle)

times = [0, 60, 180, 500]
delay = {}
for i in range(3):
    COs = data_all[i][~pd.isna(data_all[i].DOSE_2_COV)]
    COsJ = data_all[i][(data_all[i].LVAC==3)]
    INs = data_all[i][~((~pd.isna(data_all[i].DOSE_2_COV))|(data_all[i].LVAC==3))]
    for j in range(len(times)-1):
        delay['CO_m{}_'.format(times[j])+name[i]] = ((COsJ.tv1ti>=times[j])&(COsJ.tv1ti<times[j+1])).sum() + ((COs.tv2ti>=times[j])&(COs.tv2ti<times[j+1])).sum()
        delay['IN_m{}_'.format(times[j])+name[i]] = ((INs.tv1ti>=times[j])&(INs.tv1ti<times[j+1])).sum()

print('delay', delay)
