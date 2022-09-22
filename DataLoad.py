# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 19:13:10 2022

@author: TAHA
"""
import pandas as pd
import os
def load_housing_data(housing_path):
    housing_csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(housing_csv_path)