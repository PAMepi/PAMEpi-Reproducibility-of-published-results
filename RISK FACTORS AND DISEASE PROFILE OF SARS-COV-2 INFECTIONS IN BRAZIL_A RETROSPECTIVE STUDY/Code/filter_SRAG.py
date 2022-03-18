#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and Filter the OpenDataSUS SRAG database.
"""

import numpy as np
from scipy.optimize import minimize, root
import datetime
import pandas as pd


#Fixed parameters
lday = datetime.date(2021, 11, 15).strftime('%d/%m/%Y')
fday = datetime.date(2020, 2, 20).strftime('%d/%m/%Y')
max_dur = 600


def filter_srag(filename, lastday=datetime.date.today().strftime('%d/%m/%Y'),
                firstday=fday):
    data_raw = pd.read_csv(filename, delimiter=";")
    data_raw.drop_duplicates(inplace=True)
    useful_cols = ['DT_NOTIFIC', 'DT_DIGITA', 'DT_EVOLUCA', 'EVOLUCAO', 'CLASSI_FIN', 'UTI', \
                   'DT_ENTUTI', 'DT_SAIDUTI', 'DT_INTERNA', 'SG_UF_NOT', 'DT_ENCERRA', 'VACINA_COV', \
                   'DOSE_1_COV', 'DOSE_2_COV', 'CO_MUN_RES', 'NU_IDADE_N', 'SG_UF_INTE', \
                   'CS_SEXO', 'DT_SIN_PRI', 'DT_COLETA', 'DT_CO_SOR', 'PUERPERA',\
                   'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 'HEPATICA', 'ASMA',\
                   'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 'RENAL',
                   'OBESIDADE', 'OUT_MORBI', 'CS_RACA', 'LAB_PR_COV' ]
    for col in useful_cols:
        if col not in data_raw.columns:
            data_raw[col] = np.nan
    data_raw = data_raw[useful_cols]
    data_raw = data_raw[(data_raw.CLASSI_FIN == 5)]
    #adjusting in
    data_raw.loc[pd.isna(data_raw.DT_INTERNA),'DT_INTERNA'] = data_raw.DT_ENTUTI[pd.isna(data_raw.DT_INTERNA)]
    data_raw.loc[pd.isna(data_raw.DT_INTERNA),'DT_INTERNA'] = data_raw.DT_NOTIFIC[pd.isna(data_raw.DT_INTERNA)]    
    data_raw.loc[pd.isna(data_raw.DT_ENTUTI)&(data_raw.UTI==1),'DT_ENTUTI'] = data_raw.DT_INTERNA[pd.isna(data_raw.DT_ENTUTI)&(data_raw.UTI==1)]
    #adjusting out
    #UTI
    data_raw.loc[pd.isna(data_raw.DT_SAIDUTI)&(~pd.isna(data_raw.DT_ENTUTI)),'DT_SAIDUTI'] = data_raw.DT_EVOLUCA[pd.isna(data_raw.DT_SAIDUTI)&(~pd.isna(data_raw.DT_ENTUTI))]
    data_raw.loc[pd.isna(data_raw.DT_SAIDUTI)&(~pd.isna(data_raw.DT_ENTUTI)),'DT_SAIDUTI'] = data_raw.DT_ENCERRA[pd.isna(data_raw.DT_SAIDUTI)&(~pd.isna(data_raw.DT_ENTUTI))]
    #EVOL
    data_raw.loc[pd.isna(data_raw.DT_EVOLUCA),'DT_EVOLUCA'] = data_raw.DT_ENCERRA[pd.isna(data_raw.DT_EVOLUCA)]
    data_raw.loc[pd.isna(data_raw.DT_EVOLUCA), 'DT_EVOLUCA'] = data_raw.DT_SAIDUTI[pd.isna(data_raw.DT_EVOLUCA)]
    
    used_cols = ['DT_NOTIFIC', 'DT_INTERNA', 'DT_EVOLUCA', 'DT_ENTUTI', 'DT_SAIDUTI', 'EVOLUCAO', 'SG_UF_INTE',\
                 'VACINA_COV', 'DOSE_1_COV', 'DOSE_2_COV', 'CO_MUN_RES', 'NU_IDADE_N', \
                 'CS_SEXO', 'DT_SIN_PRI', 'DT_COLETA', 'DT_CO_SOR','PUERPERA',\
                 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN', 'HEPATICA', 'ASMA',\
                 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE', 'RENAL',
                 'OBESIDADE', 'OUT_MORBI', 'CS_RACA', 'LAB_PR_COV']
    data = data_raw[used_cols]
    lastday_pd = pd.to_datetime(lastday, format='%d/%m/%Y')
    firstday_pd = pd.to_datetime(firstday, format='%d/%m/%Y')
    for col in data.columns:
        if (col[:2] == 'DT') or (col[:4] == 'DOSE'):
            data.loc[:,col] = pd.to_datetime(data[col], format='%d/%m/%Y', errors='coerce')
            data = data[~(data[col]>lastday_pd)]
            data = data[~(data[col]<firstday_pd)]
    return data


#Load the OpenDataSUS 2020 and 2021 SRAG database
srag2020 = filter_srag('../Data/INFLUD20-15-11-2021.csv', lastday=lday)
print('2020 done')
srag2021 = filter_srag('../Data/INFLUD21-15-11-2021.csv', lastday=lday)
print('2021 done')
data_init = srag2020.append(srag2021, ignore_index=True)


#Calculate ICU and Hospitalization Durations
data_init['UTI_dur'] = (data_init.DT_SAIDUTI-data_init.DT_ENTUTI).dt.days
data_init['HOSP_dur'] = (data_init.DT_EVOLUCA-data_init.DT_INTERNA).dt.days
data_init = data_init[(data_init.UTI_dur<max_dur)|pd.isna(data_init.UTI_dur)]
data_init = data_init[(data_init.HOSP_dur<max_dur)]


print('adding BDI')
ibp = pd.read_csv('../Data/pop_ibp.csv')
ibp['temp'] = ibp.Cod // 10
data_init['ibp'] = np.nan
for x, valor in zip(ibp.temp, ibp.ip_vl_n):
    data_init.loc[data_init.CO_MUN_RES == x,'ibp'] = valor

#Export Data
data_init.to_csv('../Data/SRAG_filtered_morb.csv', index=False)

#%%

# Print some descriptive statistics from the SRAG database
ld = pd.to_datetime(lday, format='%d/%m/%Y')
fd = pd.to_datetime(fday, format='%d/%m/%Y')
print('Number of Confirmed: {}'.format(len(data_init)))
print('ICU: {}'.format((~pd.isna(data_init.DT_ENTUTI)).sum()))
print('ICU - Death: {}'.format(((~pd.isna(data_init.DT_ENTUTI))&(data_init.EVOLUCAO==2)).sum()))
print('ICU - Cure: {}'.format(((~pd.isna(data_init.DT_ENTUTI))&(data_init.EVOLUCAO==1)).sum()))
print('ICU - Other: {}'.format(((~pd.isna(data_init.DT_ENTUTI))&(~((data_init.EVOLUCAO==2)|(data_init.EVOLUCAO==1)))).sum()))
print('Regular Bed: {}'.format((pd.isna(data_init.DT_ENTUTI)).sum()))
print('Regular Bed - Death: {}'.format(((pd.isna(data_init.DT_ENTUTI))&(data_init.EVOLUCAO==2)).sum()))
print('Regular Bed - Cure: {}'.format(((pd.isna(data_init.DT_ENTUTI))&(data_init.EVOLUCAO==1)).sum()))
print('Regular Bed - Other: {}'.format(((pd.isna(data_init.DT_ENTUTI))&(~((data_init.EVOLUCAO==2)|(data_init.EVOLUCAO==1)))).sum()))