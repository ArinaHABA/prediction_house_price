import pandas as pd

data = pd.read_csv("../../data/geocode_msk/id_from_0_to_20000.csv")

data = data.loc[:10000]

print(data.shape)

data.to_csv("../../data/geocode_msk/id_from_0_to_10000.csv")