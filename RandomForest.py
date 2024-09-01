import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv(
    r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\complete_data.csv")

data.sort_values(["month_id","merchant_id"],ascending = [True,True],inplace = True)

willChange = data[["merchant_id","month_id","net_payment_count"]].copy()

split_point = 202305
train_data = willChange[willChange['month_id'] < split_point].copy()
val_data = willChange[willChange['month_id'] >= split_point].copy()


train_data['next_month'] = train_data.groupby("merchant_id")['net_payment_count'].shift(-1)
val_data['next_month'] = val_data.groupby("merchant_id")['net_payment_count'].shift(-1)

train_data['lag_1'] = train_data.groupby("merchant_id")['net_payment_count'].shift(1)
val_data['lag_1'] = val_data.groupby("merchant_id")['net_payment_count'].shift(1)

train_data['diff_1'] = train_data.groupby("merchant_id")['net_payment_count'].diff(1)
val_data['diff_1'] = val_data.groupby("merchant_id")['net_payment_count'].diff(1)

train_data["mean_4"] = train_data.groupby("merchant_id")['net_payment_count'].rolling(3).mean().reset_index(level=0, drop=True)
val_data["mean_4"] = val_data.groupby("merchant_id")['net_payment_count'].rolling(3).mean().reset_index(level=0, drop=True)

train_data = train_data.dropna(subset=['next_month'])
val_data = val_data.dropna(subset=['next_month'])

val_data[val_data["merchant_id"] == 2].head(100)

imputer = SimpleImputer()
train_X = imputer.fit_transform(train_data[['net_payment_count', 'lag_1', 'diff_1', 'mean_4']])
train_y = train_data['next_month']

model = RandomForestRegressor()
model.fit(train_X, train_y)

val_X = imputer.transform(val_data[['net_payment_count', 'lag_1', 'diff_1', 'mean_4']])
val_y = val_data['next_month']

predictions = model.predict(val_X)

print(mean_absolute_error(val_y,predictions))