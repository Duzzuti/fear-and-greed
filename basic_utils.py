import pandas as pd
import numpy as np

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