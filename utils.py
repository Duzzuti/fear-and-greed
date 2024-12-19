import os
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as web

from yf_exception_download import downloadCompleteHandler

# convert start_date and end_date to datetime objects set end_date to today if None
def convert_to_dates(start_date, end_date):
    if end_date == None:
        end_date = dt.date.today()
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    return start_date, end_date

# Fetch data from Yahoo Finance
def fetch_yf_data(ticker, data_dir, start_date, end_date=None, ignore_no_data=False, load_full_data=False):
    check_repl = not load_full_data
    start_date, end_date = convert_to_dates(start_date, end_date)
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
                new_data = downloadCompleteHandler(ticker, start=start_date, end=old_data_start, ignore_no_data=ignore_no_data, check_replacement_first=check_repl)
                data = pd.concat([new_data, data])
                changed = True
                new_start = start_date
            if end_date > old_data_end:
                # load new data and add to the end of the old data
                new_data = downloadCompleteHandler(ticker, start=old_data_end + dt.timedelta(days=1), end=end_date + dt.timedelta(days=1), ignore_no_data=ignore_no_data, check_replacement_first=check_repl)
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
    data = downloadCompleteHandler(ticker, start=start_date, end=end_date + dt.timedelta(days=1), ignore_no_data=ignore_no_data, check_replacement_first=check_repl)
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
    data.index = pd.to_datetime(data.index).date
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    return data

def get_sp500_possible_replacements(data_dir):
    sp500df = get_repo_data("sp500_companies.csv")
    companies = {}
    for index, row in sp500df.iterrows():
        for ticker in row['tickers'].split(","):
            if ticker in companies:
                continue
            # get last occurrence of the ticker (if it exists in the splitted list)
            last_occurrence = sp500df.index[sp500df['tickers'].apply(lambda x: ","+ticker+"," in x or x.endswith(","+ticker) or x.startswith(ticker+","))].max()
            # get the last date of that ticker (the entry after the last occurrence)
            if last_occurrence == sp500df.index[-1]:
                companies[ticker] = None
            else:
                # get entry from last occurence
                last_entry = list(sp500df.loc[last_occurrence].str.split(","))[0]
                # get entry from the next occurence
                replacement_entry = list(sp500df.loc[sp500df.index[sp500df.index.get_loc(last_occurrence) + 1]].str.split(","))[0]
                # which tickers are new in replacement_entry
                new_tickers = set(replacement_entry) - set(last_entry)
                companies[ticker] = ",".join(new_tickers)
    #sort the dict keys
    companies = dict(sorted(companies.items()))
    df = pd.DataFrame(companies, index=[0])
    # transpose the dataframe
    df = df.T
    df.to_csv(data_dir + "sp500_possible_replacements.csv")

def load_sp500_data(data_dir, start=None, load_full_data=False):
    # get ticker, start, end df
    # get yf data for each ticker and desired time frame
    # take one year of data for each ticker earlier than start_date
    sp500df = get_repo_data("sp500_companies.csv")
    if start != None:
        start = pd.to_datetime(start).date()
        tmp = sp500df[sp500df.index <= start]
        if not tmp.empty:
            sp500df = sp500df[tmp.index[-1]:]
    if not os.path.exists(data_dir + "sp500/"):
        os.mkdir(data_dir + "sp500/")
    companies = set()
    for index, row in sp500df.iterrows():
        for ticker in row['tickers'].split(","):
            if ticker in companies:
                continue
            companies.add(ticker)
            # get last occurrence of the ticker (if it exists in the splitted list)
            last_occurrence = sp500df.index[sp500df['tickers'].apply(lambda x: ","+ticker+"," in x or x.endswith(","+ticker) or x.startswith(ticker+","))].max()
            # get the last date of that ticker (the entry after the last occurrence)
            if last_occurrence == sp500df.index[-1]:
                last_date = None
            else:
                last_date = sp500df.index[sp500df.index.get_loc(last_occurrence) + 1] - pd.Timedelta(days=1)
            # get one year of data for each ticker earlier than start_date
            while True:
                try:
                    fetch_yf_data(ticker, data_dir + "sp500/", index - pd.Timedelta(days=365) , last_date, ignore_no_data=True, load_full_data=load_full_data)
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    continue
                break

def update_date_count(data_dir, repo_path="repoData/", load_full_data=False):
    # create a dictionary with the date as key and the number of csv files that contain the date as value
    try:
        if load_full_data:
            raise FileNotFoundError()
        old_date_count = pd.read_csv(repo_path + "date_count.csv")
        last_date = old_date_count["Date"].iloc[-1]
    except FileNotFoundError:
        print("Loading full data...")
        last_date = "1996-01-01"
        old_date_count = pd.DataFrame(columns=["Date", "Count"])
    # remove 366 days, because we first increment later
    first_update = (pd.to_datetime(last_date).date() - pd.DateOffset(days=366)).date()

    df_list = []
    # iterate over all csv files in test_data/sp500 folder
    for file in os.listdir(data_dir + "sp500/"):
        if file.endswith(".csv"):
            # read csv file
            df = pd.read_csv(f"{data_dir}sp500/{file}", header=[0,1], index_col=0)
            if pd.to_datetime(df.index[-1]).date() < first_update:
                continue
            df_list.append(df)

    today = pd.Timestamp.today().date()
    date_dict = {}

    # iterate over all csv files in test_data/sp500 folder
    while first_update < today:
        first_update += pd.DateOffset(days=1)
        first_update = first_update.date()
        date_dict[first_update] = 0
        for df in df_list:
            file_start_date = df.index[0]
            file_end_date = df.index[-1]
            if pd.to_datetime(file_start_date).date() <= first_update <= pd.to_datetime(file_end_date).date():
                # search for an entry with the current date
                if first_update.strftime("%Y-%m-%d") in df.index:
                    date_dict[first_update] += 1
        if date_dict[first_update] == 0:
            date_dict.pop(first_update)

    last_index = len(old_date_count)
    for date, count in date_dict.items():
        # replace old entry with new entry
        last_index += 1
        old_date_count = old_date_count[old_date_count["Date"] != date.strftime("%Y-%m-%d")]
        old_date_count.loc[last_index] = [date.strftime("%Y-%m-%d"), count]

    old_date_count.sort_values(by="Date", inplace=True)
    old_date_count.to_csv(repo_path + "date_count.csv", index=False)                


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