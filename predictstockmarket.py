import pandas as pd
import numpy as np
from datetime import datetime

# Read in dataframe
df = pd.read_csv("sphist.csv")
# Convert date column to datetime type
df["Date"] = pd.to_datetime(df["Date"])
# Sanity check that dates can be compared as boolean
df["date_compare"] = df["Date"] > datetime(year=2015, month=4, day=1)
# Sort dataframe by descending date
df = df.sort_values(by = "Date", ascending = True)
# Compute the average price from the past 5 days
df["day_5"] = df["Close"].rolling(window=5).mean()
df["day_5"] = df["day_5"].shift(1)
# Compute the average price from the past 30 days
df["day_30"] = df["Close"].rolling(window=30).mean()
df["day_30"] = df["day_30"].shift()
# Compute the average price from the past 365 days
df["day_365"] = df["Close"].rolling(window=365).mean()
df["day_365"] = df["day_365"].shift()
# Compute the ratio between average price for the past 5 days to the average price for the past 365 days
df["day_5 to day_365"] = df["day_5"]/df["day_365"]
# Remove rows with missing historical data based on past year window
df = df[df["Date"] > datetime(year=1951, month=1, day=3)]
# Drop rows with NaN values
df = df.dropna(axis = 0)

# Generate two dataframes for training and testing
train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]
# Initialize an instance of the LinearRegreggion class
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lr = LinearRegression()
features = ["day_5", "day_30", "day_365"]
# Train a linear regression model using the train Dataframe
lr.fit(train[features], train["Close"])
# Make predictions for the Close column of the test data
prediction = lr.predict(test[features])
# Compute the error between the predictions and the Close column
mae = mean_absolute_error(test["Close"], prediction)
print(mae)

