import os
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

# Fetch data from Yahoo Finance
def fetch_yf_data(ticker, data_dir, start_date, end_date=dt.date.today()):
    # convert start_date and end_date to datetime objects
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    # check if data is already saved to csv file
    # check if there is a file which ends with "_{ticker}.csv"
    for file in os.listdir(data_dir):
        if file.endswith(f"_{ticker}.csv"):
            # check whether we need to download new data
            data = pd.read_csv(data_dir + file, index_col=0, parse_dates=True, header=[0,1])
            changed = False
            old_data_start = dt.datetime.strptime(file.split("_")[0], "%Y-%m-%d").date()
            old_data_end = dt.datetime.strptime(file.split("_")[1], "%Y-%m-%d").date()
            if start_date < old_data_start:
                # load new data and add to the beginning of the old data
                new_data = yf.download(ticker, start=start_date, end=old_data_start)
                data = pd.concat([new_data, data])
                changed = True
            if end_date > old_data_end:
                # load new data and add to the end of the old data
                new_data = yf.download(ticker, start=old_data_end + dt.timedelta(days=1), end=end_date + dt.timedelta(days=1))
                data = pd.concat([data, new_data])
                changed = True
            if changed:
                data.to_csv(data_dir + f"{start_date}_{end_date}_{ticker}.csv")
                # delete old file
                os.remove(data_dir + file)
            # return data from start_date to end_date
            return data.loc[start_date:end_date]
    # if no file found, download new data
    data = yf.download(ticker, start=start_date, end=end_date + dt.timedelta(days=1))
    if data.empty:
        # open replacement list and check if ticker is in it
        with open("replacement_list.csv", "r") as f:
            replacements = f.readlines()
        for replacement in replacements:
            old, new = replacement.strip().split(",")
            if old == ticker:
                print(f"Fetching data for {new} instead of {old}...")
                data = yf.download(new, start=start_date, end=end_date + dt.timedelta(days=1))
                if data.empty:
                    print(f"Error: No data fetched for {new}. Change replacement list.")
                    exit()
        if data.empty:
            print(f"Error: No data fetched for {ticker}. Add to replacement list.")
            new_ticker = input("Enter new ticker: ")
            with open("replacement_list.csv", "a") as f:
                f.write(f"{ticker},{new_ticker}\n")
            print("Replacement list updated. Please restart the program.")
            exit()
    # save data to csv file
    data.to_csv(data_dir + f"{start_date}_{end_date}_{ticker}.csv")
    return data

def linear_weighted_backoff(metric, add, window=1000, min_backoff=0.5, max_backoff=0.5, reverse_max=None):
    """
    Normalize data with a linear weighted back-off for the max value.

    Parameters:
    - metric: A pandas Series of the data to normalize.
    - window: The lookback window size for the linear weighting.

    Returns:
    - A pandas Series with normalized values (0-100).
    """
    if reverse_max != None:
        metric = -metric
        metric = metric + reverse_max

    # need to get rid of negative values
    metric = metric + add
    if metric.min().iloc[0] < 0:
        print("Negative values in metric: " + str(metric.min().iloc[0]))
        exit()
    # Generate weights (linearly decreasing)
    weights = np.arange(1, window + 1)  # [1, 2, ..., window]
    # Normalize to min_backoff-1
    max_weights = max_backoff + (1 - max_backoff) * weights / weights.max()
    min_weights = 1 / (min_backoff + (1 - min_backoff) * weights / weights.max())
    
    def weighted_max(series):
        return max(series * max_weights[-len(series):])

    def weighted_min(series):
        return min(series * min_weights[-len(series):])

    # Apply rolling weighted max and min
    rolling_max = metric.rolling(window=window, min_periods=1).apply(weighted_max, raw=False)
    rolling_min = metric.rolling(window=window, min_periods=1).apply(weighted_min, raw=False)

    # Clip the metric to the dynamic caps
    metric_clipped = metric.clip(lower=rolling_min, upper=rolling_max)

    # Normalize between 0 and 100 using the rolling caps
    normalized = (metric_clipped - rolling_min) / (rolling_max - rolling_min) * 100

    return normalized