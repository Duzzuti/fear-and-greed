import os
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as web

from yf_exception_download import downloadCompleteHandler

# TODO fail if:
# - note that the sp500 companies are not always the same and therefore our sp500 strength and breadth are wrong (cooked)

# Fetch data from Yahoo Finance
def fetch_yf_data(ticker, data_dir, start_date, end_date=None):
    if end_date == None:
        end_date = dt.date.today()
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
            new_start = old_data_start
            new_end = old_data_end
            if start_date < old_data_start:
                # load new data and add to the beginning of the old data
                new_data = downloadCompleteHandler(ticker, start=start_date, end=old_data_start)
                data = pd.concat([new_data, data])
                changed = True
                new_start = start_date
            if end_date > old_data_end:
                # load new data and add to the end of the old data
                new_data = downloadCompleteHandler(ticker, start=old_data_end + dt.timedelta(days=1), end=end_date + dt.timedelta(days=1))
                data = pd.concat([data, new_data])
                changed = True
                new_end = end_date
            if changed:
                data.to_csv(data_dir + f"{new_start}_{new_end}_{ticker}.csv")
                # delete old file
                os.remove(data_dir + file)
            # return data from start_date to end_date
            return data.loc[start_date:end_date]
    # if no file found, download new data
    data = downloadCompleteHandler(ticker, start=start_date, end=end_date + dt.timedelta(days=1))      
    # save data to csv file
    data.to_csv(data_dir + f"{start_date}_{end_date}_{ticker}.csv")
    return data

# TODO add file support
def fetch_fred_data(name, data_dir, start_date, end_date=None):
    if end_date == None:
        end_date = dt.date.today()
    # convert start_date and end_date to datetime objects
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    # check if data is already saved to csv file
    # check if there is a file which ends with "_{ticker}.csv"
    for file in os.listdir(data_dir):
        if file.endswith(f"_{name}.csv"):
            # check whether we need to download new data
            data = pd.read_csv(data_dir + file, index_col=0, parse_dates=True, header=[0])
            changed = False
            old_data_start = dt.datetime.strptime(file.split("_")[0], "%Y-%m-%d").date()
            old_data_end = dt.datetime.strptime(file.split("_")[1], "%Y-%m-%d").date()
            new_start = old_data_start
            new_end = old_data_end
            if start_date < old_data_start:
                # load new data and add to the beginning of the old data
                new_data = web.DataReader(name, "fred", start_date, old_data_start - dt.timedelta(days=1))
                data = pd.concat([new_data, data])
                changed = True
                new_start = start_date
            if end_date > old_data_end:
                # load new data and add to the end of the old data
                new_data = web.DataReader(name, "fred", old_data_end + dt.timedelta(days=1), end_date)
                data = pd.concat([data, new_data])
                changed = True
                new_end = end_date
            if changed:
                data.to_csv(data_dir + f"{new_start}_{new_end}_{name}.csv")
                # delete old file
                os.remove(data_dir + file)
            # return data from start_date to end_date
            return data.loc[start_date:end_date]
    # if no file found, download new data
    data = web.DataReader(name, "fred", start_date, end_date)      
    # save data to csv file
    data.to_csv(data_dir + f"{start_date}_{end_date}_{name}.csv")
    return data

def get_repo_data(file_name, start_date=None, end_date=None, dir="repoData/"):
    data = pd.read_csv(dir + file_name, index_col=0, parse_dates=True)
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    return data

def fetch_sp500companies_data(data_dir, sp500_dir, start_date, end_date=dt.date.today()):
    def load_data_for_all_companies(start_date, end_date):
        sp500_tickers = pd.read_csv(data_dir + "sp500_companies.csv", header=0)
        sp500_dict = {}
        count = 0
        for ticker in sp500_tickers["Symbol"]:
            print(f"({count}/500) Fetching data for {ticker}...")
            sp500_dict[ticker] = fetch_yf_data(ticker, data_dir + sp500_dir, start_date, end_date)
            count += 1
        if count < 500:
            print("Error: Not all S&P 500 companies were fetched.")
            exit()
        return pd.concat(sp500_dict, axis=1)

    # convert start_date and end_date to datetime objects
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    

    # check if data is already saved to csv file
    # check if there is a file which ends with "_sp500.csv"
    for file in os.listdir(data_dir):
        if file.endswith(f"_sp500.csv"):
            # check whether we need to download new data
            data = pd.read_csv(data_dir + file, index_col=0, parse_dates=True, header=[0,1,2,3])
            changed = False
            old_data_start = dt.datetime.strptime(file.split("_")[0], "%Y-%m-%d").date()
            old_data_end = dt.datetime.strptime(file.split("_")[1], "%Y-%m-%d").date()
            if start_date < old_data_start:
                # load new data and add to the beginning of the old data
                new_data = load_data_for_all_companies(start_date, old_data_start - dt.timedelta(days=1))
                data = pd.concat([new_data, data])
                changed = True
            if end_date > old_data_end:
                # load new data and add to the end of the old data
                new_data = load_data_for_all_companies(old_data_end + dt.timedelta(days=1), end_date)
                data = pd.concat([data, new_data])
                changed = True
            if changed:
                data.to_csv(data_dir + f"{start_date}_{end_date}_{ticker}.csv")
                # delete old file
                os.remove(data_dir + file)
            # return data from start_date to end_date
            return data.loc[start_date:end_date]

    # if no file found, download new data
    data = load_data_for_all_companies(start_date, end_date)
    # save data to csv file
    data.to_csv(data_dir + f"{start_date}_{end_date}_sp500.csv")
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

def difference_to_ema(metric, window=125, steepness=1, reverse=False):
    """
    Calculate the difference between a metric and its Exponential Moving Average (EMA).

    Parameters:
    - metric: A pandas Series of the data to normalize.
    - window: The lookback window size for the EMA.

    Returns:
    - A pandas Series with the difference between the metric and its EMA.
    """
    if reverse:
        metric = -metric
    ema = metric.ewm(span=window, adjust=False).mean()
    return (np.tanh((metric - ema) * steepness) + 1) * 50

def pct_difference_to_ema(metric, window=125, steepness=1, reverse=False):
    """
    Calculate the percentage difference between a metric and its Exponential Moving Average (EMA).

    Parameters:
    - metric: A pandas Series of the data to normalize.
    - window: The lookback window size for the EMA.

    Returns:
    - A pandas Series with the percentage difference between the metric and its EMA.
    """
    if reverse:
        metric = -metric
    ema = metric.ewm(span=window, adjust=False).mean()
    return (np.tanh(((metric - ema)/ abs(ema))* steepness) + 1) * 50

def normalize_tanh(metric, steepness=1, shift=0, reverse=False):
    """
    Normalize data with a tanh function.

    Parameters:
    - metric: A pandas Series of the data to normalize.

    Returns:
    - A pandas Series with normalized values (0-100).
    """
    if reverse:
        metric = -metric
    return (np.tanh((metric + shift) * steepness) + 1) * 50