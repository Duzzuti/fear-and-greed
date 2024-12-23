import pandas as pd
import numpy as np
import os
import pandas_datareader.data as web

from yf_exception_download import downloadCompleteHandler

# convert start_date and end_date to datetime objects set end_date to today if None
def convert_to_dates(start_date, end_date):
    if end_date == None:
        end_date = pd.Timestamp.today().date()
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    return start_date, end_date

def get_repo_data(file_name, start_date=None, end_date=None, dir="repoData/"):
    start_date, end_date = convert_to_dates(start_date, end_date)
    data = pd.read_csv(dir + file_name, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index).date
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    return data

# Fetch data from Yahoo Finance
def fetch_yf_data(ticker, data_dir, start_date, end_date=None, ignore_no_data=False, load_full_data=False, reload_last_date=True, trading_days=None):
    check_repl = not load_full_data
    start_date, end_date = convert_to_dates(start_date, end_date)
    # check if data is already saved to csv file
    # check if there is a file which ends with "_{ticker}.csv"
    for file in os.listdir(data_dir):
        if file.endswith(f"_{ticker}.csv"):
            # check whether we need to download new data
            data = pd.read_csv(data_dir + file, index_col=0, parse_dates=True, header=[0,1])
            changed = False
            old_data_start = pd.to_datetime(file.split("_")[0]).date()
            old_data_end = pd.to_datetime(file.split("_")[1]).date()
            new_start = old_data_start
            new_end = old_data_end
            if start_date < old_data_start:
                # load new data and add to the beginning of the old data
                new_data = downloadCompleteHandler(
                    ticker, 
                    start=start_date, 
                    end=old_data_start, 
                    ignore_no_data=ignore_no_data, 
                    check_replacement_first=check_repl,
                    trading_days=trading_days
                )
                if not new_data.empty:
                    data = pd.concat([new_data, data])
                changed = True
                new_start = start_date
            if end_date > old_data_end or (end_date >= old_data_end and reload_last_date):
                # load new data and add to the end of the old data
                if reload_last_date:
                    new_data = downloadCompleteHandler(
                        ticker, 
                        start=old_data_end, 
                        end=(end_date + pd.DateOffset(days=1)).date(), 
                        ignore_no_data=ignore_no_data, 
                        check_replacement_first=check_repl,
                        trading_days=trading_days
                    )
                else:
                    new_data = downloadCompleteHandler(
                        ticker, 
                        start=(old_data_end + pd.DateOffset(days=1)).date(), 
                        end=(end_date + pd.DateOffset(days=1)).date(), 
                        ignore_no_data=ignore_no_data, 
                        check_replacement_first=check_repl,
                        trading_days=trading_days
                    )
                if not new_data.empty:
                    # remove last row if it is the same as the first row of the new data
                    if new_data.index[0] == data.index[-1]:
                        data = data.iloc[:-1]
                    data = pd.concat([data, new_data])
                changed = True
                new_end = end_date
            if changed:
                data.to_csv(data_dir + f"{new_start}_{new_end}_{ticker}.csv")
                # delete old file
                if file != f"{new_start}_{new_end}_{ticker}.csv":
                    os.remove(data_dir + file)
            # return data from start_date to end_date
            return data.loc[start_date:end_date]
    # if no file found, download new data
    data = downloadCompleteHandler(
        ticker, 
        start=start_date, 
        end=(end_date + pd.DateOffset(days=1)).date(), 
        ignore_no_data=ignore_no_data, 
        check_replacement_first=check_repl,
        trading_days=trading_days
    )
    # skip data if it is invalid (empty replacement entry)
    if type(data) != pd.DataFrame:
        if data == None:
            return None
        else:
            raise Exception(data)
    # save data to csv file
    data.to_csv(data_dir + f"{start_date}_{end_date}_{ticker}.csv")
    return data

# Fetch data from FRED
def fetch_fred_data(name, data_dir, start_date, end_date=None):
    start_date, end_date = convert_to_dates(start_date, end_date)
    # check if data is already saved to csv file
    # check if there is a file which ends with "_{ticker}.csv"
    for file in os.listdir(data_dir):
        if file.endswith(f"_{name}.csv"):
            # check whether we need to download new data
            data = pd.read_csv(data_dir + file, index_col=0, parse_dates=True, header=[0])
            changed = False
            old_data_start = pd.to_datetime(file.split("_")[0]).date()
            old_data_end = pd.to_datetime(file.split("_")[1]).date()
            new_start = old_data_start
            new_end = old_data_end
            if start_date < old_data_start:
                # load new data and add to the beginning of the old data
                new_data = web.DataReader(name, "fred", start_date, (old_data_start - pd.DateOffset(days=1)).date())
                data = pd.concat([new_data, data])
                changed = True
                new_start = start_date
            if end_date > old_data_end:
                # load new data and add to the end of the old data
                new_data = web.DataReader(name, "fred", (old_data_end + pd.DateOffset(days=1)).date(), end_date)
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