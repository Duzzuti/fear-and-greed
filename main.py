import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import scipy.stats

import put_call         # fetches the put/call ratios from alphalerts.com

data_dir = "data/"
sp500_dir = "sp500/"
downloads_dir = data_dir + "downloads/"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(data_dir + sp500_dir):
    os.makedirs(data_dir + sp500_dir)

if not os.path.exists(downloads_dir):
    os.makedirs(downloads_dir)

# Scrape S&P 500 companies
put_call_data = put_call.get_put_call_ratios(data_dir)



# Fetch close price data from Yahoo Finance
def fetch_data(ticker, start_date, dir=data_dir):
    # check if data is already saved to csv file
    try:
        data = pd.read_csv(dir + f"{start_date}_{ticker}.csv", index_col=0, parse_dates=True, header=[0,1])
        return data
    except FileNotFoundError:
        data = yf.download(ticker, start=start_date)
        if data.empty:
            # open replacement list and check if ticker is in it
            with open("replacement_list.csv", "r") as f:
                replacements = f.readlines()
            for replacement in replacements:
                old, new = replacement.strip().split(",")
                if old == ticker:
                    print(f"Fetching data for {new} instead of {old}...")
                    data = yf.download(new, start=start_date)
                    if data.empty:
                        print(f"Error: No data fetched for {new}. Change replacement list.")
                        exit()
            if data.empty:
                print(f"Error: No data fetched for {ticker}. Add to replacement list.")
                new_ticker = input("Enter new ticker: ")
                with open("replacement_list.csv", "a") as f:
                    f.write(f"{ticker},{new_ticker}\n")
                exit()
        # save data to csv file with start_date and ticker as filename
        data.to_csv(dir + f"{start_date}_{ticker}.csv")
        return data

def fetch_sp500companies_data(start_date):
    # try to read data from csv file
    try:
        data = pd.read_csv(data_dir + f"{start_date}_sp500.csv", index_col=0, parse_dates=True, header=[0,1,2,3])
        return data
    except FileNotFoundError:
        sp500_tickers = pd.read_csv(data_dir + "sp500_companies.csv", header=0)
        sp500_dict = {}
        count = 0
        for ticker in sp500_tickers["Symbol"]:
            print(f"({count}/500) Fetching data for {ticker}...")
            data = fetch_data(ticker, start_date, data_dir + sp500_dir)
            if len(data) < 10:
                print(f"Error: No data fetched for {ticker}.")
                continue
            sp500_dict[ticker] = data
            count += 1
        if count < 500:
            print("Error: Not all S&P 500 companies were fetched.")
            exit()
        sp500_data = pd.concat(sp500_dict, axis=1)
        sp500_data.to_csv(data_dir + f"{start_date}_sp500.csv")
        return sp500_data

def fetch_consumer_sentiment_data():
    # Fetch sentiment data from fred
    consumer_sentiment_data = web.DataReader("UMCSENT", "fred", start_date)

def fetch_investor_sentiment_data(rand_format : pd.DataFrame):
    # Fetch sentiment data from AAII
    # 1. Check for file in data directory
    # TODO validate the dates of the data
    # TODO scrape the recent data from the website (https://www.aaii.com/sentimentsurvey/sent_results)
    date_parser = lambda x: pd.to_datetime(x, format="%m-%d-%Y", errors='coerce')
    try:
        aaii_sentiment = pd.read_csv(data_dir + "aaii_sentiment.csv", index_col=0, parse_dates=True, date_parser=date_parser)
        
    except:
        # 2. Look in downloads directory
        try:
            aaii_sentiment = pd.read_excel(downloads_dir + "sentiment.xls", index_col=0, parse_dates=True, date_parser=date_parser)
            # only keep the 6th column
            aaii_sentiment = aaii_sentiment.iloc[:, 5]
            # remove all rows after the first NaN in index column
            aaii_sentiment = aaii_sentiment.iloc[4:pd.Series(aaii_sentiment.index.isna()[4:]).idxmax() + 4]
            # add all dates to the index
            format = rand_format.copy()
            format["AAII Sentiment"] = aaii_sentiment
            aaii_sentiment = format[["AAII Sentiment"]]
            # ffll NaN values and remove the first nan values
            aaii_sentiment.ffill(inplace=True)
            aaii_sentiment.replace(np.nan, 0, inplace=True)
            # normalize with tanh
            aaii_sentiment = ((np.tanh(aaii_sentiment * 3) + 1) / 2) * 100
            return aaii_sentiment
        except Exception as e:
            print(e)
            print("Error: No AAII sentiment data found. Please download the 'sentiment.xls' file from AAII and place it in the 'downloads' directory.")
            exit()

# Calculate market momentum
def calculate_momentum(data, moving_avg_period=125):
    moving_avg = data.rolling(window=moving_avg_period).mean()
    momentum = (data - moving_avg) / moving_avg
    return momentum * 100  # Convert to percentage

def calculate_stock_price_strength(data : pd.DataFrame, period=10):
    data_adj_close = data.xs('Adj Close', axis=1, level=1)
    # period is the number of days to look back for a 52 week high
    # 1. Calculate the 52 week highs and lows
    year_highs = data_adj_close.rolling(window=252, min_periods=1).max()
    year_lows = data_adj_close.rolling(window=252, min_periods=1).min()
    # 2. Calculate the high and low of the last period days
    period_highs = data_adj_close.rolling(window=period, min_periods=1).max()
    period_lows = data_adj_close.rolling(window=period, min_periods=1).min()
    # 3. Calculate the number of stocks that are at their 52 week high and low
    num_highs = ((data_adj_close == period_highs) & (data_adj_close == year_highs)).sum(axis=1)
    num_lows = ((data_adj_close == period_lows) & (data_adj_close == year_lows)).sum(axis=1)
    # 4. Calculate the stock price strength
    stock_price_strength = num_highs / (num_highs + num_lows) * 100
    # replace NaN values with 50 (no stocks at 52 week high or low)
    stock_price_strength.fillna(50, inplace=True)
    return stock_price_strength

def calculate_breadth(data : pd.DataFrame, period=10):
    # Prepare a DataFrame to store results
    breadth_results = []

    # Process each ticker
    for ticker in data.columns.get_level_values(0).unique():
        df = data[ticker].copy()
        
        # Calculate the daily price change
        df['Change'] = df['Adj Close'].diff()
        
        # Define volumes for "up" and "down" days
        df['Volume Up'] = df['Volume'].where(df['Change'] > 0, 0)  # Volume for "up" days
        df['Volume Down'] = df['Volume'].where(df['Change'] <= 0, 0)  # Volume for "down" days

        # Store the results with dates for aggregation
        breadth_results.append(df[['Volume Up', 'Volume Down']])

    # Combine all tickers' data
    combined_breadth = pd.concat(breadth_results).groupby(level=0).sum()
    # Calculate total breadth
    res = combined_breadth['Volume Up'] / (combined_breadth['Volume Down'] + combined_breadth['Volume Up'])
    res.fillna(0.5, inplace=True)
    res = pd.DataFrame(res, columns=['Breadth Ratio'], index=combined_breadth.index)
    return res.ewm(span=period).mean() * 100

def calculate_put_call_ratios(start_date):
    # TODO missing data for some dates
    # extend put_call_data to start date by adding NaN values
    tmp_put_call_data = put_call_data.copy()
    start_date = pd.to_datetime(start_date)
    # add missing dates to the put_call_data
    idx = pd.date_range(start_date, tmp_put_call_data.index[0])
    for date in idx:
        if date not in tmp_put_call_data.index:
            tmp_put_call_data.loc[date] = np.nan
    # sort the index
    tmp_put_call_data.sort_index(inplace=True)
    tmp_put_call_data.ffill(inplace=True)
    # fill 0 values with 1
    tmp_put_call_data.replace(0, tmp_put_call_data.mean(), inplace=True)
    return tmp_put_call_data

def calculate_save_haven(sp500_data : pd.DataFrame, t10ybond_data : pd.DataFrame, period=20, bond_weight=None):
    # calculate the period returns of the stock market
    sp500_period_return = sp500_data.pct_change(period)
    # calculate the return per annum
    sp500_annual_return = ((1 + sp500_period_return) ** (252 / period) - 1) * 100
    # calculate the difference between the first columns
    if bond_weight == None:
        bond_weight = 252 / period
    sp500_annual_return["Diff"] = sp500_annual_return["^SP500TR"] - t10ybond_data["^TNX"] * bond_weight
    return sp500_annual_return[["Diff"]]

def calculate_junk_bond_spread(start_date, rand_format : pd.DataFrame):
    # Fetch data from FRED
    junk_bond_spread : pd.DataFrame = web.DataReader("BAMLH0A0HYM2", "fred", start_date)  # junk bond spread
    junk_bond_spread.ffill(inplace=True)
    # rename the index to "Date"
    junk_bond_spread.index.name = "Date"
    rand_format = rand_format.copy()
    rand_format["BAMLH0A0HYM2"] = junk_bond_spread
    # remove the other columns
    junk_bond_spread = rand_format[["BAMLH0A0HYM2"]]
    print(junk_bond_spread)
    return junk_bond_spread

def normalize_metric(metric):
    return (metric - metric.min()) / (metric.max() - metric.min()) * 100

def z_score(metric):
    mean = metric.mean()
    std = metric.std()
    return (metric - mean) / std

def normalize_z_score(metric):
    z_scores = z_score(metric)
    # Scale to 0-100
    scaled = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min()) * 100
    return scaled

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


# Example: Combine normalized metrics
def calculate_fear_greed_index(*normalized_metrics):
    return pd.concat(normalized_metrics, axis=1).mean(axis=1)


# Fetch data for S&P 500
start_date = '2000-01-01'  # Adjust to cover at least 20 years
sp500_data = fetch_data('^SP500TR', start_date)["Close"]
vix_data = fetch_data('^VIX', start_date)["Close"]
sp500companies_data = fetch_sp500companies_data(start_date)
t10ybond_data = fetch_data('^TNX', start_date)["Close"]
aaii_sentiment = fetch_investor_sentiment_data(sp500_data)

# Calculate metrics
sp500_momentum = calculate_momentum(sp500_data)
sp500_strength = calculate_stock_price_strength(sp500companies_data, period=15)
sp500_breadth = calculate_breadth(sp500companies_data, period=10)
put_call_ratios = calculate_put_call_ratios(start_date)
safe_haven_demand = calculate_save_haven(sp500_data, t10ybond_data, period=50)
junk_bond_spread = calculate_junk_bond_spread(start_date, safe_haven_demand)

# Normalize and combine into a single index
normalized_momentum = linear_weighted_backoff(sp500_momentum, 100, window=1000, min_backoff=0.8, max_backoff=0.85)
normalized_vix = linear_weighted_backoff(vix_data, 0, window=1000, min_backoff=0.7, max_backoff=0.95, reverse_max=100)
normalized_put_call = linear_weighted_backoff(put_call_ratios, 0, window=1000, min_backoff=0.4, max_backoff=0.9, reverse_max=3).ewm(span=5).mean()
normalized_safe_haven_demand = linear_weighted_backoff(safe_haven_demand, 300, window=1000, min_backoff=0.95, max_backoff=0.75)
normalized_junk_bond_spread = linear_weighted_backoff(junk_bond_spread, 0, window=1000, min_backoff=0.7, max_backoff=0.95, reverse_max=25)#.ewm(span=5).mean()

fear_greed_index = calculate_fear_greed_index(normalized_momentum, normalized_vix, sp500_strength, sp500_breadth, normalized_put_call, normalized_safe_haven_demand, normalized_junk_bond_spread, aaii_sentiment)
fear_greed_index2 = fear_greed_index.ewm(span=20).mean()

# Plot the Fear and Greed Index
plt.figure(figsize=(12, 6))
#plt.plot(aaii_sentiment.index, aaii_sentiment, label="10Y Bond Yield", color="blue")
#plt.plot(normalized_safe_haven_demand.index, normalized_safe_haven_demand, label="10Y Bond Yield (Normalized)", color="green")
plt.plot(fear_greed_index.index, fear_greed_index, label="Fear & Greed Index", color="purple")
#plt.plot(fear_greed_index2.index, fear_greed_index2, label="Fear & Greed Index (Smoothed)", color="orange")
plt.title("Fear & Greed Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.axhline(50, color='grey', linestyle='--', label="Neutral")
plt.legend()
plt.show()
