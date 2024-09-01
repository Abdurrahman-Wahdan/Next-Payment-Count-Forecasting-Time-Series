from sklearn.metrics import mean_absolute_error
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

data = pd.read_csv(
    r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\train.csv")
data1 = pd.read_csv(
    r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\complete_data2.csv")

data["merchant_id"] = data["merchant_id"].apply(lambda x: x[9:])

for name in data.columns[[2, 3, 4, 6]]:
    data[name] = data[name].apply(lambda x: x[-1])

data["mcc_id"] = data["mcc_id"].apply(lambda x: x[4:])

data.sort_values(by=["merchant_id", "month_id"], ascending=[True, False], inplace=True)

grouped_data = data.groupby("merchant_id")

missing_data = []

for (merchant, group) in grouped_data:
    first_data = group.head(1).copy()

    if (202309 - first_data["month_id"].iloc()[0] > 3):
        missing_data.append(merchant)

missing_data = [int(x) for x in missing_data]

data1 = data1[~data1['merchant_id'].isin(missing_data)]

grouped_data = data1.groupby("merchant_id")

predictions = []

for merchant, group in tqdm(grouped_data, desc="Generating Forecasts"):
    merchant_data = group[(group['month_id'] < 202307) & (group['month_id'] > 202112)]

    df = merchant_data[['month_id', 'net_payment_count']].copy()
    df.columns = ['ds', 'y']

    df['ds'] = pd.to_datetime(df['ds'], format='%Y%m')
    df = df.set_index('ds')

    # ARIMA model
    order = (0, 1, 2)  # You can experiment with the order parameter
    model = ARIMA(df['y'], order=order)
    results = model.fit()

    # Forecasting
    forecast_steps = 3
    forecast = results.get_forecast(steps=forecast_steps)

    # Extracting predictions
    forecast_values = forecast.predicted_mean[-forecast_steps:].to_list()
    forecast_values.insert(0, merchant)

    predictions.append(forecast_values)



#TEST MAE
#--------------------------------------------------------------------------------------
    
data = pd.read_csv(r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\complete_data.csv")

data = data[~data['merchant_id'].isin(missing_data)]

grouped_data = data.groupby("merchant_id")

targets = []

for merchant, group in grouped_data:
    target = group[(group['month_id'] > 202306)]["net_payment_count"].to_list()
    targets.append(target)

flat_list1 = [item for sublist in predictions for item in sublist[1:]]
flat_list2 = [item for sublist in targets for item in sublist]    
flat_list2 = [max(0, round(number)) for number in flat_list2]

print(mean_absolute_error(flat_list1, flat_list2))


#SUBMISSION
#-----------------------------------------------------------------------
    
# submission = pd.read_csv(r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\sample_submission.csv")

# for i in tqdm(range(len(predictions)), desc= "Preparing Submission File"):
#     merchant_id = str(predictions[i][0])

#     # Update for the month 202310
#     submission.loc[submission["id"] == "202310merchant_" + merchant_id, "net_payment_count"] = max(0, round(predictions[i][1]))

#     # Update for the month 202311
#     submission.loc[submission["id"] == "202311merchant_" + merchant_id, "net_payment_count"] = max(0, round(predictions[i][2]))

#     # Update for the month 202312
#     submission.loc[submission["id"] == "202312merchant_" + merchant_id, "net_payment_count"] = max(0, round(predictions[i][3]))

# submission.to_csv(r'C:\Users\furkan\Masaüstü\Projects\Python\Project1\output\predictions.csv', index=False)