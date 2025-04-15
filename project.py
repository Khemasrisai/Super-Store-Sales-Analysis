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

#Scatter-Plot
mlt.figure(figsize=(8, 6))
sb.scatterplot(data=df, x='Discount', y='Profit', hue='Category')
mlt.title('Discount vs Profit')
mlt.xlabel('Discount')
mlt.ylabel('Profit')
mlt.tight_layout()
mlt.show()

#Line-Graph
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)
monthly_sales = df['Sales'].resample('ME').sum()

mlt.plot(monthly_sales, marker='o', color='blue')
mlt.title('Monthly Sales Trend')
mlt.xlabel('Month')
mlt.ylabel('Total Sales')


#Pair-Plot
data = df[['Sales', 'Profit', 'Discount', 'Region']]
sb.pairplot(data, hue='Region')
mlt.show()

#Box-Plot
mlt.figure(figsize=(12, 6))
sb.boxplot(x='Sub-Category', y='Sales', data=df)
mlt.title('Sales Distribution by Sub-Category')
mlt.xticks(rotation=45)
mlt.tight_layout()
mlt.show()


#Combination of Bar-Plot and Line-Plot
category_sales = df.groupby('Category')['Profit'].sum().reset_index()

mlt.figure(figsize=(8, 5))
sb.barplot(x='Category', y='Profit', data=category_sales, color="lightgreen")
sb.lineplot(x='Category', y='Profit', data=category_sales, color="green", marker='o')
mlt.title('Profit by Category')
mlt.xlabel('Category')
mlt.ylabel('Total Profit')
mlt.show()


#OBJECTIVE 4 - (Outliers And Anomaly Detection)
#Correlation
columns_to_check = ['Sales', 'Discount', 'Profit', 'Quantity']
selected_df = df[columns_to_check]
# Correlation
c = selected_df.corr()
print("Correlation:\n", c)

print("\n")

# Covariance
cv = selected_df.cov()
print("Covariance: \n", cv)

print("\n")

# Outlier detection using IQR
Q1 = selected_df.quantile(0.25)
Q3 = selected_df.quantile(0.75)
IQR = Q3 - Q1

# Lower and Upper Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("\nLower Bound:\n", lower_bound)
print("\nUpper Bound:\n", upper_bound)

print("\n")

outlier = selected_df[(selected_df < lower_bound) | (selected_df > upper_bound)].sum()
print("\nOutliers:\n", outlier)

print("\n")

sb.boxplot(data=selected_df)
mlt.title("Boxplot: Sales, Discount, Profit, Quantity")

print("\n")

from scipy import stats
score=stats.zscore(df[columns_to_check],nan_policy='omit') 
outliers=(abs(score>3)).sum(axis=0)
print(outliers)

mlt.show()
