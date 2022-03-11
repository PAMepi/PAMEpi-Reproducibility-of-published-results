#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:11:46 2021

@author: lhunlindeion
"""

import numpy as np
from scipy.optimize import minimize, root
import datetime
import pandas as pd

ref = datetime.date(2019, 12, 31)

data_init = pd.read_csv('SRAG_filtered_morb.csv')
data_init['MORTE'] = (data_init.EVOLUCAO == 2)
states = np.r_[np.array([ 'BR' ]), data_init.SG_UF_INTE.unique()]
for col in data_init.columns:
    if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
        data_init.loc[:,col] = pd.to_datetime(data_init[col], format='%Y/%m/%d', errors='coerce')
data_init['tth'] = (data_init.DT_INTERNA - data_init.DT_SIN_PRI).dt.days
data_init['ti'] = (data_init.DT_EVOLUCA - data_init.DT_INTERNA).dt.days

# data_init['ttn'] = (data_init.DT_NOTIFIC - data_init.DT_SIN_PRI).dt.days
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


# ibp = pd.read_csv('data-cidacs_ipb_municipios.csv')
# ibpv = [ibp.ip_vl_n[ibp.ip_qntl_n==i].min() for i in range(1,6)]
# ibpv = ibpv + [ibp.ip_vl_n.max()]
# ibpv = np.linspace(data_init.ibp.min(), data_init.ibp.max(), 6)

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

def rec_region(states):
    regioes = {'SC':'S', 'PR':'S', 'RS':'S',
               'RJ':'SE', 'MG':'SE', 'ES':'SE', 'SP':'SE',
               'DF':'CO', 'MT':'CO', 'MS':'CO', 'GO':'CO',
               'AP':'N', 'PA':'N', 'TO':'N', 'AM':'N', 'RO':'N', 'RR':'N', 'AC':'N',
               'BA':'NE', 'AL':'NE', 'SE':'NE', 'PE':'NE', 'PB':'NE', 'RN':'NE',\
               'CE':'NE', 'PI':'NE', 'MA':'NE'}
    reg = states.map(regioes)
    return reg

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
data_init['reg'] = rec_region(data_init.SG_UF_INTE)
data_init = data_init[~pd.isna(data_init.CS_RACA)]
data_init = data_init[data_init.CS_RACA!=9]
data_init = data_init[data_init.CS_SEXO != 'I']
data_init['SEXO'] = (data_init.CS_SEXO == 'M')

gr_risco = ['PNEUMOPATI', 'IMUNODEPRE', 'OBESIDADE', 'SIND_DOWN', \
            'RENAL', 'NEUROLOGIC', 'DIABETES', 'PUERPERA', 'OUT_MORBI', \
            'HEMATOLOGI', 'ASMA', 'HEPATICA', 'CARDIOPATI']
for col in gr_risco:
    data_init[col] = (data_init[col] == 1)

trad_raca = {1:'Branca', 2:'Preta', 3:'Amarela', 4:'Parda', 5:'Indigena'}

data_init['RACA'] = data_init['CS_RACA'].map(trad_raca)

col_names = ['MORTE', 'AGE_GRP', 'BDI_GRP', 'eUTI', 'NVACC', 'SEXO', 'RACA', 'TINT']#, 'reg']
col_names = col_names + gr_risco 
X = data_init[col_names]
X = X.dropna()
X, labs = create_dummy(X, ['AGE_GRP', 'RACA', 'BDI_GRP', 'TINT'], [1, 0, 1, 0] )
out = X
Y = out['MORTE'].to_numpy()
X = out.drop(['MORTE'], axis=1).to_numpy(dtype=float).T


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
    return res.x, sqr, cor

import time
st = time.time()
res_null = null_pars(Y)
res, sqr, cor = fit_logit(X,Y)
print(time.time()-st)
print('N=', X.shape[1])
#%%
print('G=', 2*(res_null[1]-nll_logit(X,Y,res)))
saida = {'variable': [], 'value':[], 'std':[], 'z_val':[], 'odd':[], 'sum_col':[]}
for name, val, std, z in zip(out.columns, res, sqr, np.abs(res)/sqr):
    saida['variable'].append(name)
    saida['value'].append(val)
    saida['std'].append(std)
    saida['z_val'].append(z)
    saida['odd'].append(np.exp(val))
    saida['sum_col'].append(out[name].sum())

with open('corr_logit.csv', 'w') as f:
    f.write(' ,' + ', '.join(out.columns) + '\n')
    for i, line in enumerate(cor):
        f.write(out.columns[i] + ', ' + ', '.join([str(ele) for ele in line.tolist()])+ '\n')
    

sa = pd.DataFrame(saida)
sa.to_csv('logit_results_ag_tint.csv', index=False)    