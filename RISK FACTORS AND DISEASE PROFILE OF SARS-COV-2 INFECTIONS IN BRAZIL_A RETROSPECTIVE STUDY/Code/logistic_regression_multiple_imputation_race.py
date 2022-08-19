#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the logistic regression for the mortality of hospitalized patients
using multiple imputation method to address missing ethnic/racial background
entries.
"""

import numpy as np
from scipy.optimize import root
import datetime
import pandas as pd
import joblib
import random

ref = datetime.date(2019, 12, 31)

data_init = pd.read_csv('../Data/SRAG_filtered_morb.csv')
data_init['MORTE'] = (data_init.EVOLUCAO == 2)
states = np.r_[np.array([ 'BR' ]), data_init.SG_UF_INTE.unique()]
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')
data_init['tth'] = (data_init.DT_INTERNA - data_init.DT_SIN_PRI).dt.days
data_init['ti'] = (data_init.DT_EVOLUCA - data_init.DT_INTERNA).dt.days

data_init.loc[pd.isna(data_init.VACINA_COV),'VACINA_COV'] = 0

ages = [0, 18, 30, 40, 50, 65, 75, 85, np.inf]
nsep = len(ages) - 1
data_init['AGE_GRP'] = ''
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i]),'AGE_GRP'] = 'AG85+'
    else:
        data_init.loc[(data_init.NU_IDADE_N>=ages[i])&(data_init.NU_IDADE_N<ages[i+1]), 'AGE_GRP'] = 'AG{}t{}'.format(ages[i],ages[i+1])
data_init = data_init[data_init.AGE_GRP != '']

tis = [0, 4, 12, 40, np.inf]
nsep = len(tis) - 1
data_init['TINT'] = ''
for i in range(nsep):
    if i == nsep-1:
        data_init.loc[(data_init.ti>=tis[i]),'TINT'] = 'TM40'
    else:
        data_init.loc[(data_init.ti>=tis[i])&(data_init.ti<tis[i+1]), 'TINT'] = 'T{}t{}'.format(tis[i],tis[i+1])
data_init = data_init[data_init.TINT != '']



ibpv = [data_init.ibp.quantile(x) for x in [0.0,0.2,0.4,0.6,0.8,1.0]]
names = [ 'BDI_' + i for i in ['0', '1', '2', '3', '4']]
data_init['BDI_GRP'] = ''
for i in range(5):
    if i == 4:
        data_init.loc[(data_init.ibp>=ibpv[i]),'BDI_GRP'] = names[i]
    else:
        data_init.loc[(data_init.ibp>=ibpv[i])&(data_init.ibp<ibpv[i+1]), 'BDI_GRP'] = names[i]
data_init = data_init[data_init.AGE_GRP != '']
data_init = data_init[data_init.BDI_GRP != '']
#%%

def generate_label_dummy(labels):
    labels = [str(x) for x in labels]
    init = labels[0]
    return ['{}_{}'.format(init, x) for x in labels[1:]]

def create_dummy(x, cols, sort=None):
    y = x[cols]
    output = x.drop(cols, axis=1)
    labels = list()
    if type(sort) == type(None):
        sort = [False] * len(cols)
    for cname, needsort in zip(y.columns, sort):
        z = y[cname]
        labs = z.unique()
        if needsort == 1:
            labs.sort()
        elif needsort == -1:
            labs.sort()
            labs = labs[::-1]
        elif needsort == 'D' or needsort == 'C':
            counts = np.array([ (z==val).sum() for val in labs])
            labs = labs[counts.argsort()]
            if needsort == 'D':
                labs = labs[::-1]
        nl = len(labs)
        zt = np.nan * np.ones((len(z)))
        for i in range(nl):
            zt[z==labs[i]] = i
        if nl > 10:
            raise ValueError("Number of categorical values must be smaller than 10")
        else:
            new_series = np.zeros((nl-1, len(z)))
            for i in range(1,nl):
                new_series[i-1,zt==i] = 1
            new_series[0, np.isnan(zt)] = np.nan
        label_tt = generate_label_dummy(labs)
        for ser, ltt in zip(new_series, label_tt):
            output[ltt] = ser
        labels = labels + label_tt
    return output, labels

def logit(x, pars):
    if len(pars) > 1:
        P = pars[0] + (pars[1:].reshape((-1,1)) * x).sum(axis=0)
    else:
        P = pars[0]
    return (1. / (1. + np.exp(-P))).flatten()

def nll_logit(x, y, pars):
    p = logit(x, pars)
    return - (y * np.log(p) + (1-y) * np.log(1-p)).sum()

data_init['eUTI'] = ~pd.isna(data_init.DT_ENTUTI)
data_init['NVACC'] = (data_init.VACINA_COV!=1)
data_init = data_init[data_init.CS_SEXO != 'I']
data_init['SEXO'] = (data_init.CS_SEXO == 'M')

gr_risco = ['PNEUMOPATI', 'IMUNODEPRE', 'OBESIDADE', 'SIND_DOWN', \
            'RENAL', 'NEUROLOGIC', 'DIABETES', 'PUERPERA', 'OUT_MORBI', \
            'HEMATOLOGI', 'ASMA', 'HEPATICA', 'CARDIOPATI']
for col in gr_risco:
    data_init[col] = (data_init[col] == 1)

trad_raca = {1:'Branca', 2:'Preta', 3:'Amarela', 4:'Parda', 5:'Indigena', 9:'Desconhecida'}

data_init['RACA'] = data_init['CS_RACA'].map(trad_raca)
data_init.loc[pd.isna(data_init.RACA), 'RACA'] = 'Desconhecida' 
col_names = ['MORTE', 'AGE_GRP', 'BDI_GRP', 'eUTI', 'NVACC', 'SEXO', 'RACA', 'TINT']
col_names = col_names + gr_risco 
X = data_init[col_names]
X0 = X.dropna()


#%%
def null_pars(y):
    par = np.log(y.sum()/(1-y).sum())
    nll = nll_logit([], y, [par])
    return par, nll

def der_eqs(x, y, par):
    p = logit(x, par)
    eqs = [(y-p).sum()]
    for xi in x:
        eqs.append((xi*(y-p)).sum())
    return eqs


def info_logit(x, pars):
    n = len(pars)
    p = logit(x, pars)
    p = p * (1 - p)
    xi = np.r_[np.ones((1, len(x[0]))), x]
    jac = np.empty((n,n))
    for i in range(n):
        for j in range(i,n):
            jac[i,j] = (xi[i]*xi[j]*p).sum()
            jac[j,i] = jac[i,j]
    return jac

def var_logit(x, pars):
    jac = info_logit(x, pars)
    return np.linalg.inv(jac)

def fit_logit(x, y):
    x0 = np.zeros(x.shape[0]+1)
    res = root(lambda p: der_eqs(x,y,p), x0,  method='hybr', jac=lambda p: -info_logit(x,p))
    var = var_logit(x, res.x)
    sqr = np.sqrt(np.diag(var))
    cor = var / (sqr.reshape((1,-1))*sqr.reshape((-1,1)))
    return res.x, sqr*sqr, cor

def input_race(data, frac, col, key_to_fill):
    Ntf = (data[col] == key_to_fill).sum()
    new_vals = pd.DataFrame(np.random.rand(Ntf), columns=['random'])
    lim = 0
    new_vals['col'] = ''
    for ref in frac.keys():
        new_vals.loc[(new_vals.random>=lim)&(new_vals.random<=lim + frac[ref]), 'col'] = ref
        lim = lim + frac[ref]
    data_new = data.copy()
    data_new.loc[data_new[col] == key_to_fill, col] = new_vals.col.to_list()
    return data_new

def input_race_v2(data, frac, col):
    Ntf = len(data)
    data['random'] = np.random.rand(Ntf).tolist()
    lim = 0
    for ref in frac.keys():
        data.loc[(data.random>=lim)&(data.random<=lim + frac[ref]), col] = ref
        lim = lim + frac[ref]
    return data



def mult_imput_random(data, ref_col, missing_col, missing_label):
    saida = data.copy()
    ref = data[data[missing_col] != missing_label]
    while len(saida[saida[missing_col] == missing_label]) > 0:
        to_imput = saida[saida[missing_col] == missing_label].iloc[0]
        bool_to_replace = saida[ref_col+[missing_col]].eq(to_imput[ref_col+[missing_col]]).all(1)
        Ni = bool_to_replace.sum()
        saida.loc[bool_to_replace, missing_col] = random.choices(ref.loc[ref[ref_col].eq(to_imput[ref_col]).all(1),missing_col].to_list(), k=Ni)
    return saida



_, nll = null_pars(X0.MORTE.to_numpy())
XX = X0[X0.RACA!='Desconhecida']
out, labs = create_dummy(XX, ['AGE_GRP', 'RACA', 'BDI_GRP', 'TINT'], ['D', "D", 1, 0] )
saida = {name:[] for name in out.columns}
for name in out.columns:
    saida[name+'_variance'] = []
Y = out['MORTE'].to_numpy()
X = out.drop(['MORTE'], axis=1).to_numpy(dtype=float).T
res, var, cor = fit_logit(X,Y)
for val, varr, name in zip(res, var, out.columns):
    saida[name].append(val)
    saida[name+'_variance'].append(varr)
saida['G'] = [2*(nll-nll_logit(X,Y,res))]



def calculate_logit(Xi, nll=nll):
    sa = {}
    X = mult_imput_random(Xi, ['AGE_GRP', 'BDI_GRP', 'eUTI'], 'RACA', 'Desconhecida')
    out, labs = create_dummy(X, ['AGE_GRP', 'RACA', 'BDI_GRP', 'TINT'], ['D', "D", 1, 0] )
    Y = out['MORTE'].to_numpy()
    x = out.drop(['MORTE'], axis=1).to_numpy(dtype=float).T
    try:
        res, var, cor = fit_logit(x,Y)
        sa['G'] = 2*(nll-nll_logit(x,Y,res))
    except:
        res = np.nan * np.ones(len(out.columns))
        var = np.nan * np.ones(len(out.columns))
        sa['G'] = np.inf
    for val, varr, name in zip(res, var, out.columns):
        sa[name] = val
        sa[name+'_variance'] = varr
    return sa

nj = 32
all_res = joblib.Parallel(n_jobs=4, verbose=15)(joblib.delayed(calculate_logit)(X0,) for i in range(nj))


for res in all_res:
    for key in res.keys():
        saida[key].append(res[key])
    

sa = pd.DataFrame(saida)
sa.to_csv('../Results/logit_results_imputation.csv', index=False)