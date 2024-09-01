# Next Payment Count Forecasting (Time Series Analysis)

**Since this is a hobby project, it was not written with a clean code approach. Moreover, of the studies mentioned below, only some files from the works have been added to the repo.**

This project was prepared for a datathon organized by Iyzico. Some information about merchants and the number of net payments for 45 months for each were given and we were asked to forecast the net payment count of the next 3 months.

Since it was a time series problem, we thought that the correlation between the data was not highly significant. We still did the analysis, external data was also added, studied with time series models such as `SARIMA` that work with additional data and seasonal data, but in the end we decided on no correlation.

In the given dataset, not all merchants had 45 months of data. Total net payment count for each merrchant were available between 0 and 45. Therefore, first, we had to fill the missing data with an algorithm and we decided to fill the data in a linear interpolation manner.

For example, if 2 monthly data is missing between 10 and 40, then:

> (max - min) / (missing_data_count + 1) = (40 - 10) / (2 + 1) = 10
>
> We can add +10 to the left-handed data for each missing data. 
> 
> So, the new data will be "10, 20, 30, 40" which is linear.

After interpolationing the missing data, we experimented some different time series models such as `ARIMA`, `SARIMA`, `PROPHET`, `LSTM` and got the best MAE scores with `ARIMA`. Therefore, we performed hyperparameter tuning with `ARIMA`.

Moreover, to turn the problem into a classic machine learning problem, we created a new dataset using shifting the time series data (-1 and +1) and tried some `gradient boosting models`, `linear regression` and `RandomForest regressor`. 

Still, we finished the project with `ARIMA` because `ARIMA` gave us the best score at the end.
