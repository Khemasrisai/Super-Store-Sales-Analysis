import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt
import seaborn as sb


df = pd.read_csv("C:\\Users\\giris\\Documents\\python project\\superstore-dataset-py-pro.csv")
print(df)


#Summary of this dataset
print(df.head())
print(df.describe())
print(df.info())
print(df.nunique())



#Data Cleaning
print(df.isnull().sum())
print(df.dropna())




