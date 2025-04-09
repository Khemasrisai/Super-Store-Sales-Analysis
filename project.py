import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt
import seaborn as sb


df = pd.read_csv("C:\\Users\\giris\\Documents\\python project\\superstore-dataset-py-pro.csv")
print(df)

#OBJECTIVE 1

#Data Overview
print(df.shape())
print(df.columns())
print(df.head())
print(df.info())
print(df.nunique())

#Data Cleaning
print(df.isnull().sum().sum())
print(df.dropna())
print(df.drop(["Ship Mode","Segment","Postal Code","Product ID","Order ID"],axis=1))


#OBJECTIVE 2
max_profit = df['Profit'].max()  
print("Max Profit:",max)

min_profit = df['Profit'].min()  
print("Max Profit:",min)

loss_rows = df[df['Profit'] < 0]
print(loss_rows)


numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
correlation = df[numeric_cols].corr()

# Plot the heatmap
mlt.figure(figsize=(8, 6))
sb.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=5)
mlt.title('Correlation Heatmap: Sales, Quantity, Discount, Profit')
mlt.show()




#OBJECTIVE 3
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
