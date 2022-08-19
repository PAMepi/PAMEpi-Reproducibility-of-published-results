#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the resulting mortality outcome odd ratios from the multiple imputation
logistic regression. This script uses the outcome from 
logistic_regression_multiple_imputation_race.py file.
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

def calculate_IC(values, inter_variances):
    mul = 1.96
    m_val = np.mean(values)
    var = inter_variances.mean() + (1 + 1/len(values)) * values.var(ddof=1)
    return m_val, var, m_val - mul * np.sqrt(var), m_val + mul * np.sqrt(var)



data = pd.read_csv('../Results/logit_results_imputation.csv')

#Remove the complete case fit
data = data.iloc[1:]
#Remove the parameters of fits that did not converge
data = data[np.isfinite(data.G)]
data = data[data.G>4.1e5]

# plt.ion()
# plt.figure()
# plt.hist(data.G )


name ={'MORTE': 'Constant',
        'eUTI': 'ICU',
        'NVACC': 'Unvaccinated',
        'SEXO': 'Male',
        'PNEUMOPATI': 'Lung Dis.',
        'IMUNODEPRE': 'Immunodef.',
        'OBESIDADE': 'Obesity',
        'SIND_DOWN': 'Down Syn.',
        'RENAL': 'Kidney Dis.',
        'NEUROLOGIC': 'Neurol. Dis.',
        'DIABETES': 'Diabetes',
        'PUERPERA': 'Postpartum',
        'OUT_MORBI': 'Other Comor.',
        'HEMATOLOGI': 'Hematol. Dis.',
        'ASMA': 'Asthma',
        'HEPATICA': 'Liver Dis.',
        'CARDIOPATI': 'Heart Dis.',
        'AG50t65_AG65t75': 'Age 65-74',
        'AG50t65_AG40t50': 'Age 40-49',
        'AG50t65_AG75t85': 'Age 75-84',
        'AG50t65_AG30t40': 'Age 30-39',
        'AG50t65_AG85+': 'Age 85+',
        'AG50t65_AG18t30': 'Age 18-29',
        'AG50t65_AG0t18': 'Age 0-17',
        'Branca_Parda': 'Ethn. Mixed', 
        'Branca_Preta': 'Ethn. Black',
        'Branca_Amarela': 'Ethn. Asian',
        'Branca_Indigena': 'Ethn. Indigenous',
        'BDI_0_BDI_1': 'BDI 1',
        'BDI_0_BDI_2': 'BDI 2',
        'BDI_0_BDI_3': 'BDI 3', 
        'BDI_0_BDI_4': 'BDI 4',
        'T4t12_T12t40': 'Hosp. 12-39 days',
        'T4t12_T0t4': 'Hosp. 0-3 days',
        'T4t12_TM40': 'Hosp. 40+ days'}
saida = {'variable':[],
         'Mean':[],
         'CI Low':[],
         'CI High':[],
         }
for col in data.columns:
    if (col != 'G') and (col.find('_variance')<0):
        val = data[col].to_numpy()        
        var = data[col+'_variance'].to_numpy()
        saida['variable'].append(name[col])
        me, va, l1, h1 = calculate_IC(val, var)
        saida['Mean'].append(np.exp(me))
        saida['CI Low'].append(np.exp(l1))
        saida['CI High'].append(np.exp(h1))

saida = pd.DataFrame(saida)
saida.to_csv('../Results/imputed_race_table.csv', index=False)