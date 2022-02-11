#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:47:06 2021

@author: lhunlindeion
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

def median_estimate(X, CI):
    n = len(X)
    lmd = binom.ppf((1-CI)/2, n, 0.5)
    mmd = binom.ppf((1+CI)/2, n, 0.5)
    Xo = np.sort(X)
    return np.median(Xo), Xo[int(lmd)], Xo[int(mmd)-1]
    
def freq_estimate(X, CI):
    n = len(X)
    P = (X==True).sum()
    lmd = binom.ppf((1-CI)/2, n, P/n)
    mmd = binom.ppf((1+CI)/2, n, P/n)
    return P/n, lmd/n, mmd/n
    

def create_filter_cont(data, ycol, xcols, fname, col_extra=None, CI=0.95):
    lme = norm.ppf((1-CI)/2)
    mme = norm.ppf((1+CI)/2)
    data = data[~pd.isna(data[ycol])]
    saida = {'name': [], 'mean': [], 'CIme_L':[], 'CIme_H':[], 'median':[], \
             'CImd_L':[], 'CImd_H':[]}
    saida['name'].append('All')
    saida['mean'].append(np.mean(data[ycol]))
    saida['CIme_L'].append(np.mean(data[ycol]) + lme*np.std(data[ycol])/len(data[ycol]))
    saida['CIme_H'].append(np.mean(data[ycol]) + mme*np.std(data[ycol])/len(data[ycol]))
    med, cl, ch = median_estimate(data[ycol], CI)
    saida['median'].append(med)
    saida['CImd_L'].append(cl)
    saida['CImd_H'].append(ch)
    if col_extra != None:
        for val_extra in data[col_extra].unique():
            data_extra = data[data[col_extra]==val_extra]
            saida['name'].append('All_'+str(val_extra))
            saida['mean'].append(np.mean(data_extra[ycol]))
            saida['CIme_L'].append(np.mean(data_extra[ycol]) + lme*np.std(data_extra[ycol])/len(data_extra[ycol]))
            saida['CIme_H'].append(np.mean(data_extra[ycol]) + mme*np.std(data_extra[ycol])/len(data_extra[ycol]))
            med, cl, ch = median_estimate(data_extra[ycol], CI)
            saida['median'].append(med)
            saida['CImd_L'].append(cl)
            saida['CImd_H'].append(ch)
    for xcol in xcols:
        for val in data[xcol].unique():
            if val is np.nan:
                data_fil = data[pd.isna(data[xcol])]
            else:
                data_fil = data[data[xcol]==val]
            data_fil = data_fil[~pd.isna(data_fil[ycol])]
            saida['name'].append(str(xcol)+'_'+str(val))
            saida['mean'].append(np.mean(data_fil[ycol]))
            saida['CIme_L'].append(np.mean(data_fil[ycol]) + lme*np.std(data_fil[ycol])/len(data_fil[ycol]))
            saida['CIme_H'].append(np.mean(data_fil[ycol]) + mme*np.std(data_fil[ycol])/len(data_fil[ycol]))
            med, cl, ch = median_estimate(data_fil[ycol], CI)
            saida['median'].append(med)
            saida['CImd_L'].append(cl)
            saida['CImd_H'].append(ch)
            if col_extra != None:
                for val_extra in data_fil[col_extra].unique():
                    data_extra = data_fil[data_fil[col_extra]==val_extra]
                    saida['name'].append(str(xcol)+'_'+str(val)+'_'+str(val_extra))
                    saida['mean'].append(np.mean(data_extra[ycol]))
                    saida['CIme_L'].append(np.mean(data_extra[ycol]) + lme*np.std(data_extra[ycol])/len(data_extra[ycol]))
                    saida['CIme_H'].append(np.mean(data_extra[ycol]) + mme*np.std(data_extra[ycol])/len(data_extra[ycol]))
                    med, cl, ch = median_estimate(data_extra[ycol], CI)
                    saida['median'].append(med)
                    saida['CImd_L'].append(cl)
                    saida['CImd_H'].append(ch)
    saida = pd.DataFrame(saida)
    saida.to_csv(fname, index=False)
                
def create_filter_binary(data, ycol, xcols, fname, CI=0.95):
    lme = norm.ppf((1-CI)/2)
    mme = norm.ppf((1+CI)/2)
    data = data[~pd.isna(data[ycol])]
    saida = {'name': [], 'mean': [], 'CIme_L':[], 'CIme_H':[]}
    mea, cl, ch = freq_estimate(data[ycol], CI)
    saida['name'].append('All')
    saida['mean'].append(mea)
    saida['CIme_L'].append(cl)
    saida['CIme_H'].append(ch)
    for xcol in xcols:
        for val in data[xcol].unique():
            if val is np.nan:
                data_fil = data[pd.isna(data[xcol])]
            else:
                data_fil = data[data[xcol]==val]
            data_fil = data_fil[~pd.isna(data_fil[ycol])]
            mea, cl, ch = freq_estimate(data_fil[ycol], CI)
            saida['name'].append(str(xcol)+'_'+str(val))
            saida['mean'].append(mea)
            saida['CIme_L'].append(cl)
            saida['CIme_H'].append(ch)
    saida = pd.DataFrame(saida)
    saida.to_csv(fname, index=False)


ref = datetime.date(2019, 12, 31)
max_dur = 90
data0 = pd.read_csv('SRAG_filtered_morb.csv')

for col in data0.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data0.loc[:,col] = pd.to_datetime(data0[col], format='%Y/%m/%d', errors='coerce')

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
data0['AGEGRP'] = ''
for i in range(nsep):
    if i == nsep-1:
        data0.loc[(data0.NU_IDADE_N>=ages[i]),'AGEGRP'] = 'AG85+'
    else:
        data0.loc[(data0.NU_IDADE_N>=ages[i])&(data0.NU_IDADE_N<ages[i+1]), 'AGEGRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])

trad_raca = {1:'Branca', 2:'Preta', 3:'Amarela', 4:'Parda', 5:'Indigena'}

data0['RACA'] = data0['CS_RACA'].map(trad_raca)
ibpv = [data0.ibp.quantile(x) for x in [0.0,0.2,0.4,0.6,0.8,1.0]]
names = [ 'BDI' + i for i in ['0', '1', '2', '3', '4']]
data0['BDIGRP'] = ''
for i in range(5):
    if i == 4:
        data0.loc[(data0.ibp>=ibpv[i]),'BDIGRP'] = names[i]
    else:
        data0.loc[(data0.ibp>=ibpv[i])&(data0.ibp<ibpv[i+1]), 'BDIGRP'] = names[i]

gr_risco = ['PNEUMOPATI', 'IMUNODEPRE', 'OBESIDADE', 'SIND_DOWN', \
            'RENAL', 'NEUROLOGIC', 'DIABETES', 'PUERPERA', 'OUT_MORBI', \
            'HEMATOLOGI', 'ASMA', 'HEPATICA', 'CARDIOPATI']
data0['COMOR'] = 'NO'
for risco in gr_risco:
    data0.loc[data0[risco]==1,'COMOR'] = 'YES'

data0['MORTE'] = 'OTHER'
data0.loc[data0.EVOLUCAO==2, 'MORTE'] = "MORTE"
data0.loc[data0.EVOLUCAO==1, 'MORTE'] = "CURA"
#removing unknown outcomes
data0 = data0[data0.MORTE !='OTHER']


data0['VACINA'] = (data0.VACINA_COV == 1)
data0['TSM'] = (data0.DT_EVOLUCA-data0.DT_SIN_PRI).dt.days
data0.loc[data0.MORTE!="MORTE", 'TSM'] = np.nan
data0['TSH'] = (data0.DT_INTERNA-data0.DT_SIN_PRI).dt.days
data0['TSI'] = (data0.DT_ENTUTI-data0.DT_SIN_PRI).dt.days
create_filter_cont(data0, 'UTI_dur', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'ICU_dur.csv', 'MORTE' )
create_filter_cont(data0, 'HOSP_dur', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'HOSP_dur.csv', 'MORTE' )
create_filter_cont(data0, 'TSM', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'TimeSintomasMorte.csv', 'MORTE' )
create_filter_cont(data0, 'TSH', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'TimeSintomasInterna.csv', 'MORTE' )
create_filter_cont(data0, 'TSI', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'TimeSintomasICU.csv', 'MORTE' )

data_m = data0[data0.MORTE != 'OTHER']
data_m['MORTE'] = (data_m.MORTE=='MORTE')
create_filter_binary(data_m[~pd.isna(data_m.DT_ENTUTI)], 'MORTE', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'Mortalidade_ICU.csv')
create_filter_binary(data_m[pd.isna(data_m.DT_ENTUTI)], 'MORTE', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'Mortalidade_HOSP.csv')
data0['THI'] = (data0.DT_ENTUTI-data0.DT_INTERNA).dt.days
data0['DirICU'] = (data0.THI == 0)
create_filter_binary(data0, 'DirICU', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'Direct_to_ICU.csv')
dataind = data0[data0.THI != 0]
dataind['frac'] = (~pd.isna(dataind.DT_ENTUTI))
create_filter_binary(dataind, 'frac', ['AGEGRP', 'CS_SEXO', 'RACA', 'BDIGRP', 'VACINA', 'COMOR' ], 'Frac_to_ICU.csv')