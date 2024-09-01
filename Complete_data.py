import pandas as pd
from tqdm import tqdm

data = pd.read_csv(
    r"C:\Users\furkan\Masaüstü\Projects\Python\Project1\input\train.csv")

data["merchant_id"] = data["merchant_id"].apply(lambda x : x[9:])

for name in data.columns[[2, 3, 4, 6]]:
    data[name] = data[name].apply(lambda x: x[-1])

data["mcc_id"] = data["mcc_id"].apply(lambda x: x[4:])

# Convert "merchant_id" to numerical format
data['merchant_id'] = pd.to_numeric(data['merchant_id'], errors='coerce')

# Sort by "merchant_id" and "month_id"
data.sort_values(by=["merchant_id", "month_id"], ascending=[True, False], inplace=True)

all_months = pd.date_range(start='2020-01-01', end='2023-09-01', freq='MS').strftime('%Y%m').astype(int)

grouped_data = data.groupby("merchant_id")

missing_data_list = []
for (merchant, group) in tqdm(grouped_data, desc= "Creating Data"):
    missing_data_list.append([])
    
    existing_months = group["month_id"].unique()
    
    i=0
    while (i < (len(existing_months)-1)):
        
        date1 = str(group.iloc[i]["month_id"])
        date2 = str(group.iloc[i+1]["month_id"])
         
        if (i == 0):
            difference = (2023 - int(date1[0:4]))*12 + 9 - int(date1[4:])
            
            if (difference > 0):
                missing_data_list[-1].extend([round(group.iloc[i]["net_payment_count"])]*difference)     
        
        difference = (int(date1[0:4]) - int(date2[0:4]))*12 + int(date1[4:]) - int(date2[4:])    
        if (difference > 1):
            missing_data_list[-1].extend([round(group.iloc[i]["net_payment_count"] - ((group.iloc[i]["net_payment_count"] - group.iloc[i+1]["net_payment_count"])/(difference))*j) for j in range(1, difference)])
            
        if (i == len(existing_months) - 2):
            difference = ((int(date2[0:4]) - 2020)*12 + int(date2[4:]) - 1)
            
            if (difference > 0):
                missing_data_list[-1].extend([round(group.iloc[-1]["net_payment_count"])]*difference)
        
        i += 1
    else:
        if (i==0):
            date1 = str(group.iloc[i]["month_id"])
            difference = (2023 - int(date1[0:4]))*12 + 9 - int(date1[4:])
            
            if (difference > 0):
                missing_data_list[-1].extend([round(group.iloc[i]["net_payment_count"])]*difference)     
            
            difference = ((int(date1[0:4]) - 2020)*12 + int(date1[4:]) - 1)
            
            if (difference > 0):
                missing_data_list[-1].extend([round(group.iloc[-1]["net_payment_count"])]*difference)


data_list = []
for i, (merchant, group) in tqdm(enumerate(grouped_data), total = grouped_data.ngroups, desc= "Filling Data"):
    
    existing_months = group["month_id"].unique()
    missing_months = list(set(all_months) - set(existing_months))
    missing_months.sort(reverse=True)
    
    for j, month in enumerate(missing_months):
        new_data = group.head(1).copy()
        new_data["net_payment_count"] = missing_data_list[i][j]
        new_data["month_id"] = month
        data_list.append(new_data.copy())

concated_data = pd.concat(data_list, ignore_index=True)
data = pd.concat([data, concated_data], ignore_index=True)
data.sort_values(by=["merchant_id", "month_id"], ascending=[True, False], inplace=True)

data.to_csv(r'C:\Users\furkan\Masaüstü\Projects\Python\Project1\output\complete_data.csv', index=False)