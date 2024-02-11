import numpy as np
import pandas as pd
import altair as alt
from altair import Chart, Scale, Y
from dbnomics import fetch_series, fetch_series_by_api_link
import torch
from torch.utils.data import Dataset, DataLoader
import os

"""
Step 1 : Preprocess dataset
"""

main_countries_wb = ["USA", "CHN", "JPN", "DEU", "IND", "GBR", "FRA", "RUS", "CAN", "ITA", "KOR"]
main_countries_imf = ["US", "CN", "JP", "DE", "IN", "GB", "FR", "RU", "CA", "IT", "KR"]

def fetch_or_load(*args, **kwargs):
    """
    fetch data from dbnomics or load
    """
    map_countries_wb_imf = dict(zip(main_countries_wb, main_countries_imf))
    folderpath = "./dataframes/"
    filename = f"{args[0]}_{args[1]}.csv"
    os.makedirs(folderpath, exist_ok=True)
    try:
        df = pd.read_csv(folderpath+filename)
    except FileNotFoundError:
        df = fetch_series(*args, **kwargs)
        if args[0] == 'WB':
            df = df.rename(columns={"indicator": "INDICATOR",
                                    "country": "REF_AREA",
                                    "frequency": "FREQ"})
            df["REF_AREA"] = df["REF_AREA"].replace(map_countries_wb_imf)
        df.to_csv(folderpath+filename, index=False)
        print(filename, "generated")
    return df
    

def make_econ_dataframes():
    """
    call fetch_or_load 4 times
    """
    df0 = fetch_or_load("WB", "GEM", dimensions={
        "indicator": ["NYGDPMKTPSACD"],
        "country": main_countries_wb,
        "frequency": ["Q"],
    }, max_nb_series=1000)

    df1 = fetch_or_load("IMF", "BOP", dimensions={
        "INDICATOR": ["BCA_BP6_USD", "BACK_BP6_USD"],
        "REF_AREA": main_countries_imf,
        "FREQ": ["Q"],
    }, max_nb_series=1000)

    df2 = fetch_or_load("IMF", "MFS", dimensions={
        "INDICATOR": ["FIGB_PA"],
        "REF_AREA": main_countries_imf,
        "FREQ": ["M"],
    }, max_nb_series=1000)

    df3 = fetch_or_load("IMF", "CPI", dimensions={
        "INDICATOR": ["PCPI_PC_CP_A_PT"],
        "REF_AREA": main_countries_imf,
        "FREQ": ["M"],
    }, max_nb_series=1000)

    dfs = [df0, df1, df2, df3]
    for i, df in enumerate(dfs):
        # df = df[df.value.notna()]
        df['period'] = pd.to_datetime(df['period'])
        df = df[df.period >= "1976-01-01"]
        df = df[df.period <= "2021-07-01"]
        dfs[i] = df
        print(len(df))
    return dfs

"""
Step 2 : Define custom dataset and dataloader. 
"""

class EconDataset(Dataset): 
    def __init__(self, dfs):
        super().__init__()
        self.data = torch.from_numpy(dfs.to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    dfs = make_econ_dataframes()
    # concat_chart = show_dataframes(dfs, "All Dataframes")
    # dfs = concat_dataframes(dfs, )
    # dataset = EconDataset(dfs)
    # loader = DataLoader(dataset, batch_size=64, shuffle=True)
