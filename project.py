import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt
import seaborn as sb

df = pd.read_csv("C:\\Users\\giris\\Documents\\python project\\superstore-dataset-py-pro.csv")
print(df)

#OBJECTIVE 1 - (Data Overview And Cleaning)
#Data Overview
print("Shape: ",df.shape)
print("Column Names: ",df.columns)
print(df.head(10))
print("DataSet Info: ",df.info())
print(df.nunique())

print("\n")

#Data Cleaning
print("Null Values: ",df.isnull().sum().sum())
print(df.dropna())
print(df.drop(["Ship Mode","Segment","Postal Code","Product ID","Order ID"],axis=1))


#OBJECTIVE 2 (Key Metrics And Summary Statistics)
print("Describe: ",df[['Sales', 'Profit', 'Discount', 'Quantity']].describe())

print("\n")
print("Total Sales: ₹", df['Sales'].sum())
print("Total Profit: ₹", df['Profit'].sum())
print("Total Orders:", len(df))
print("Average Discount: {:.2f}%".format(df['Discount'].mean() * 100))
print("Total Quantity Sold:", df['Quantity'].sum())
print("Most Profitable Sub-Category:", df.groupby('Sub-Category')['Profit'].sum().idxmax())
print("Least Profitable Sub-Category:", df.groupby('Sub-Category')['Profit'].sum().idxmin())



#OBJECTIVE 3 - (Data Visualizations for Trends & Patterns)

#Pie-Plot
region_sales = df.groupby('Region')['Sales'].sum()
mlt.figure(figsize=(7, 7))
explode = [0.01,0.01,0.01,0.01]
mlt.pie(region_sales, labels=region_sales.index, explode=explode, autopct='%1.1f%%')
mlt.title('Sales Distribution by Region')
mlt.axis('equal')
mlt.show() 


#Bar-Plot
mlt.figure(figsize=(10, 6))
sb.barplot(x='Sub-Category', y='Profit', data=df, estimator=sum, errorbar=None)
mlt.title('Total Profit by Sub-Category')
mlt.xticks(rotation=45)
mlt.tight_layout()
mlt.show()

