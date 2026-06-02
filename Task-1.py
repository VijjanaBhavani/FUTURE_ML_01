import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
df=pd.read_csv("sales_data.csv",encoding="latin1")
print(df.head())
print(df.info())
print(df.columns)
print(df.shape)
#time based features
df['Order Date']=pd.to_datetime(df["Order Date"])
df['Year']=df["Order Date"].dt.year
df["Month"]=df["Order Date"].dt.month
df["Day"]=df["Order Date"].dt.day
print(df[["Year","Month","Day"]].head())
#trend analysis
monthly_sales=df.groupby('Month')['Sales'].sum()
print(monthly_sales)
plt.figure(figsize=(8,5))
plt.plot(monthly_sales.index,monthly_sales.values)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Trend")
plt.savefig("monthly_sales_trend.png")
plt.show()
x=df[["Year","Month","Day","Quantity","Discount","Profit"]]
y=df["Sales"]
train_size=int(len(x)*0.8)
x_train=x[:train_size]
x_test=x[train_size:]
y_train=y[:train_size]
y_test=y[train_size:]
print(x_train.shape)
print(x_test.shape)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
predictions=model.predict(x_test)
mae=mean_absolute_error(y_test,predictions)
print("MAE:",mae)
r2=r2_score(y_test,predictions)
print("R2 score:",r2)
plt.figure(figsize=(10,5))
plt.plot(y_test.values,label="Actual sales")
plt.plot(predictions,label="Predicted sales")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales Forecast")
plt.savefig("actual_vs_predicted.png")
plt.legend()
plt.show()
