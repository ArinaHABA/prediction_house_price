import numpy as np
import pandas as pd

list_id = [('0', '10000'), ('10001', '60000'),('60001', '120000'),('120001', '200000'),('200001', '300000'),('300001', '381860')]

df1 = pd.read_csv(f'../../data/geocode_msk/id_from_' + list_id[0][0] + '_to_' + list_id[0][1] + '.csv')
df2 = pd.read_csv(f'../../data/geocode_msk/id_from_' + list_id[1][0] + '_to_' + list_id[1][1] + '.csv')

df = pd.concat([df1, df2])

for i in range(2 , len(list_id)):
    df_add = pd.read_csv(f'../../data/geocode_msk/id_from_' + list_id[i][0] + '_to_' + list_id[i][1] + '.csv')
    df = pd.concat([df, df_add])

df.reset_index(drop = True, inplace = True)
df_msk = pd.read_csv("../../data/interim/msk_2018_2021_clear.csv")

print(df.shape)
print(df_msk.shape)

#df_msk = pd.concat([df_msk, df['district']], axis = 1)

df_msk['district'] = df['district']
 
df.to_csv(r'{0}.csv'.format('../../data/geocode_msk/geocode_concat'))
df_msk.to_csv(r'{0}.csv'.format('../../data/processed/msk_2018_2021'))





