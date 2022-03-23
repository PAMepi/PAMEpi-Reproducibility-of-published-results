#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the mild to moderate infections column of Table 1, it needs the 
sg.csv file filtered from the datalake

"""

import numpy as np
import datetime
import pandas as pd


data_init = pd.read_csv('../Data/IBP_classes_categories.csv')
ibp_mins = data_init.n_f.to_numpy()
ibp_maxs = data_init.n_to.to_numpy()

ibpv = np.r_[ibp_mins, ibp_maxs[-1]]


sg = pd.read_csv('../Data/sg.csv')
sg = sg[(sg.evolucaoCaso != 'Óbito' ) & (sg.evolucaoCaso != 'Internado') & (sg.evolucaoCaso != 'Internado em UTI')]
#%%
sg['municipioIBGE'] = pd.to_numeric(sg.municipioIBGE, errors='coerce')
ibp = pd.read_csv('../Data/pop_ibp.csv')
sg['ibp'] = np.nan
for x, valor in zip(ibp.Cod, ibp.ip_vl_n):
    sg.loc[sg.municipioIBGE == x,'ibp'] = valor

names = [ 'BDI_' + i for i in ['0', '1', '2', '3', '4']]
sg['BDI_GRP'] = ''
for i in range(5):
    if i == 4:
        sg.loc[(sg.ibp>=ibpv[i]),'BDI_GRP'] = names[i]
    else:
        sg.loc[(sg.ibp>=ibpv[i])&(sg.ibp<ibpv[i+1]), 'BDI_GRP'] = names[i]
#%%
gr_risco = ['RESPIRAT', 'IMUNODEPRE', 'OBESIDADE', \
            'RENAL', 'DIABETES', 'PUERPERA', \
            'CARDIOPATI','OUT_MORBI']

def find_risk(row):
    saida = [False]*8
    linha = row['condicoes']
    if type(linha) == str:
        linha = linha.lower()
        if (linha.find('respirat') >= 0) or (linha.find('pulmonar') >= 0):
            saida[0] = True
        if linha.find('imuno') >= 0:
            saida[1] = True
        if (linha.find('obesidade') >= 0) or (linha.find('obesa') >= 0):
            saida[2] = True
        if linha.find('renais') >= 0:
            saida[3] = True
        if linha.find('diabetes') >= 0:
            saida[4] = True
        if (linha.find('puérpera') >= 0) or (linha.find('puerpera') >= 0):
            saida[5] = True
        if (linha.find('cardíaca') >= 0) or (linha.find('cardiaca') >= 0) or \
            (linha.find('cardoaca') >= 0) or (linha.find('cardio') >= 0):
            saida[6] = True
        if (linha.find('outros') >= 0) or (linha.find('erisipela') >= 0) or \
            (linha.find('gestante') >= 0) or (linha.find('has') >= 0):
            saida[7] = True
    return saida

sg['RESPIRAT'], sg['IMUNODEPRE'], sg['OBESIDADE'], sg['RENAL'], sg['DIABETES'],\
    sg['PUERPERA'], sg['CARDIOPATI'], sg['OUT_MORBI'] = zip(*sg.apply(find_risk, axis=1))

conds = list()
for cond in sg.condicoes.unique():
    if type(cond) == str:
        for co in cond.split(','):
            co = co.strip(' ').lower()
            if co not in conds:
                conds.append(co)
    else:
        if cond not in conds:
            conds.append(cond)
#%%

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
sg['AGE_GRP'] = ''
for i in range(nsep):
    if i == nsep-1:
        sg.loc[(sg.idade>=ages[i]),'AGE_GRP'] = 'AG85+'
    else:
        sg.loc[(sg.idade>=ages[i])&(sg.idade<ages[i+1]), 'AGE_GRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])

sexo = {'female': (sg.sexo==1).sum(), 'male':(sg.sexo==2).sum(),
        'unknown': (sg.sexo==9).sum() + pd.isna(sg.sexo).sum()}
print(sexo)
# trad_raca = {2:'Branca', 1:'Preta', 5:'Amarela', 4:'Parda', 3:'Indigena'}
trad_raca = {4:'White', 1:'Black', 3:'Yellow', 5:'Mixed', 2:'Indigenous'}
sg['RACA'] = sg['racaCor'].map(trad_raca)
raca = {loc:(sg.RACA==loc).sum() for loc in sg.RACA.unique()}
raca[np.nan] = pd.isna(sg.RACA).sum()
print(raca)
bdi = {loc:(sg.BDI_GRP==loc).sum() for loc in sg.BDI_GRP.unique()}
print(bdi)
ages = {loc:(sg.AGE_GRP==loc).sum() for loc in sg.AGE_GRP.unique()}
print(ages)

no_risco = np.ones((len(sg)))
grupos = dict()
for risco in gr_risco:
    grupos[risco] = (sg[risco]==1).sum()
    no_risco = no_risco * (1-(sg[risco] == 1))

grupos['NENHUM'] = no_risco.sum()
print(grupos)
