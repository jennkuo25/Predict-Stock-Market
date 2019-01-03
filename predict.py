import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("sphist.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["date_compare"] = df["Date"] > datetime(year=2015, month=4, day=1)
df = df.sort_values(by = "Date", ascending = True)
print(df.head())