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
print(df.isnull().sum().sum())
print(df.dropna())


#Correlation
c=df.corr(numeric_only=True)
print(c)
sb.heatmap(c,cmap="coolwarm",annot=True,linewidth=5,fmt="0.2f")
mlt.xticks(rotation=360)
mlt.show()


#Covariance
cv=df.cov(numeric_only=True)
print(cv)



#Outliers
numeric_values=df.select_dtypes(include=['number']).columns
print(numeric_values)


Q1=df[numeric_values].quantile(0.25)
Q3=df[numeric_values].quantile(0.75)
print(Q1)
print(Q3)

IQR=Q3-Q1
print(IQR)

#lower bound
lower_bound = Q1-1.5*IQR
print(lower_bound)

#upper bound
upper_bound = Q3+1.5*IQR
print(upper_bound)



outlier=df[(df[numeric_values]<lower_bound) | (df[numeric_values]>upper_bound)]
print(outlier)

sb.boxplot(df)
mlt.show()
