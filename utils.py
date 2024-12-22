import os
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf

from yf_exception_download import downloadCompleteHandler
from scraper.aaii_scraper import scrape_aaii
from scraper.insider_scraper import scrape_insider
from scraper.margin_stats_scraper import scrape_margin_stats
from scraper.put_call_scraper import scrape_put_call
from scraper.sp500_company_scraper import scrape_companies
from metric_base import Metric
from metrics import YieldCurve, T10YearYield, JunkBondSpread, SaveHavenDemand, ConsumerSentiment, SP500Momentum, PutCallRatio, InsiderTransactions, AAIISentiment, MarginStats, VIX, StockPriceBreadth, StockPriceStrength
import basic_utils

# Fetch data from Yahoo Finance
def fetch_yf_data(ticker, data_dir, start_date, end_date=None, ignore_no_data=False, load_full_data=False, reload_last_date=True, trading_days=None):
    check_repl = not load_full_data
    start_date, end_date = basic_utils.convert_to_dates(start_date, end_date)
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
    start_date, end_date = basic_utils.convert_to_dates(start_date, end_date)
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

def get_sp500_possible_replacements(data_dir):
    sp500df = basic_utils.get_repo_data("sp500_companies.csv")
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
    update_trading_days(data_dir)
    trading_days = pd.read_csv(data_dir + "trading_days.csv", index_col=0, header=[0,1])
    sp500df = basic_utils.get_repo_data("sp500_companies.csv")
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
                last_date = (sp500df.index[sp500df.index.get_loc(last_occurrence) + 1] - pd.DateOffset(days=1)).date()
            # get one year of data for each ticker earlier than start_date
            while True:
                try:
                    fetch_yf_data(
                        ticker, 
                        data_dir + "sp500/", 
                        (index - pd.DateOffset(days=365)).date(), 
                        last_date,
                        ignore_no_data=True, 
                        load_full_data=load_full_data,
                        trading_days=trading_days
                    )
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}")
                    continue
                break

def update_date_count(data_dir, repo_path="repoData/", load_full_data=False, allow_skip=True):
    # create a dictionary with the date as key and the number of csv files that contain the date as value
    try:
        if load_full_data:
            raise FileNotFoundError()
        old_date_count = pd.read_csv(repo_path + "date_count.csv")
        last_date = old_date_count["Date"].iloc[-1]
    except FileNotFoundError:
        print("Loading full data...")
        allow_skip = False
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

    # process the last date again and if it matches, we should be able to skip older dates
    if allow_skip:
        count = 0
        skip_update_date = pd.to_datetime(last_date).date()
        for df in df_list:
            file_start_date = df.index[0]
            file_end_date = df.index[-1]
            if pd.to_datetime(file_start_date).date() <= skip_update_date <= pd.to_datetime(file_end_date).date():
                # search for an entry with the current date
                if skip_update_date.strftime("%Y-%m-%d") in df.index:
                    count += 1
        if count == old_date_count["Count"].iloc[-1]:
            first_update = skip_update_date

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

def update_trading_days(data_dir):
    if os.path.exists(data_dir + "trading_days.csv"):
        trading_days = pd.read_csv(data_dir + "trading_days.csv", index_col=0, header=[0,1])
        trading_days.index = pd.to_datetime(trading_days.index).date
        last_date = trading_days.index[-1]
        if last_date < pd.Timestamp.today().date():
            df = yf.download("^GSPC", start=last_date)
            df.index = pd.to_datetime(df.index).date
            # remove last_date from trading days
            trading_days = trading_days[trading_days.index != last_date]
            trading_days = pd.concat([trading_days, df])
            trading_days.to_csv(data_dir + "trading_days.csv")
    else:
        yf.download("^GSPC", start="1995-01-01").to_csv(data_dir + "trading_days.csv")

def update_breadth_data(sp500_dir, df_breadth, trading_days):
    last_breadth_date = df_breadth.index[-1]
    df_list = []
    # iterate over all csv files in sp500 folder
    for file in os.listdir(sp500_dir):
        if file.endswith(".csv"):
            # read csv file
            df = pd.read_csv(f"{sp500_dir}{file}", header=[0,1], index_col=0)
            if df.index[-1] < last_breadth_date.strftime("%Y-%m-%d"):
                continue
            file_start_date = file.split("_")[0]
            file_start_date = pd.to_datetime(file_start_date).date() + pd.DateOffset(days=365)
            first_relevant_date = max(file_start_date.strftime("%Y-%m-%d"), last_breadth_date.strftime("%Y-%m-%d"))
            # init date is the date that needs to be added to calculate the difference for the first date (the first date before the first relevant date in date_count_data)
            init_date = trading_days.index[(trading_days.index < first_relevant_date)][-1]
            # use date_count_data to add missing dates between the first relevant date and the last date
            df = df.reindex(trading_days.index[(trading_days.index <= df.index[-1])], fill_value=None)
            # fill any nans after the first valid value but before the last valid value
            # Identify the indices to forward-fill
            ffill_mask = df.notna() | df.bfill().notna()
            # Forward-fill the data
            df = df.ffill().where(ffill_mask)
            # remove all irrelevant data
            df = df[df.index >= init_date]
            df_list.append(df)

    # Process each ticker
    for df in df_list:
        # Calculate the daily price change
        df['Change'] = df['Adj Close'].diff()
        
        # Define volumes for "up" and "down" days
        df['Volume Up'] = df['Volume'].where(df['Change'] > 0, 0)  # Volume for "up" days
        df['Volume Down'] = df['Volume'].where(df['Change'] < 0, 0)  # Volume for "down" days

    if not df_list:
        print("No new data to process")
    else:
        # Combine all tickers' data
        combined_breadth = pd.concat(df_list).groupby(level=0).sum()
        # Calculate total breadth
        res = combined_breadth['Volume Up'] / (combined_breadth['Volume Down'] + combined_breadth['Volume Up'])
        res.fillna(0.5, inplace=True)
        res = pd.DataFrame(res, columns=['Breadth Ratio'], index=combined_breadth.index)
        # drop first row
        res = res[1:]
        res = pd.concat([df_breadth[df_breadth.index != last_breadth_date], res])
        res.to_csv("repoData/breadth_ratio.csv", index_label="Date")

def update_strength_data(sp500_dir, df_strength, trading_days):
    last_strength_date = df_strength.index[-1]
    num_highs = {}
    num_lows = {}
    trading_days.index = pd.to_datetime(trading_days.index).date
    for date in trading_days.index:
        if date >= last_strength_date:
            num_highs[date] = 0
            num_lows[date] = 0
    # iterate over all csv files in sp500 folder
    for file in os.listdir(sp500_dir):
        if file.endswith(".csv"):
            # read csv file
            df = pd.read_csv(f"{sp500_dir}{file}", header=[0,1], index_col=0, parse_dates=True)["Adj Close"]
            df.index = pd.to_datetime(df.index).date
            if df.index[-1] < last_strength_date:
                continue
            file_start_date = file.split("_")[0]
            actual_start_date = max((pd.to_datetime(file_start_date).date() + pd.DateOffset(days=365)).date(), last_strength_date)
            # add all trading days that are missing in the dataframe
            df = df.reindex(trading_days.index, axis=0)
            # Identify the indices to forward-fill
            ffill_mask = df.notna() | df.bfill().notna()
            # Forward-fill the data
            df = df.ffill().where(ffill_mask)
            df.dropna(inplace=True)
            # get row that is after actual_start_date
            first_valid_row = df[df.index >= actual_start_date].iloc[0]
            load_data_since = (first_valid_row.name - pd.DateOffset(days=365)).date()
            year_high = df.loc[df[(df.index <= first_valid_row.name) & (df.index >= load_data_since)].idxmax()].iloc[0]
            year_low = df.loc[df[(df.index <= first_valid_row.name) & (df.index >= load_data_since)].idxmin()].iloc[0]
            current_row_index = df.index.get_loc(first_valid_row.name)
            current_row = first_valid_row
            while current_row_index < len(df) and current_row.notna().all():
                if current_row.iloc[0] >= year_high.iloc[0]:
                    year_high = current_row
                elif current_row.iloc[0] <= year_low.iloc[0]:
                    year_low = current_row
                else:
                    one_year_ago = (current_row.name - pd.DateOffset(days=365)).date()
                    if year_high.name < one_year_ago:
                        # calculate year_high again where the date is less than one year ago from the current date
                        year_high = df.loc[df[(df.index >= one_year_ago) & (df.index <= current_row.name)].idxmax()].iloc[0]
                    if year_low.name < one_year_ago:
                        # calculate year_low again
                        year_low = df.loc[df[(df.index >= one_year_ago) & (df.index <= current_row.name)].idxmin()].iloc[0]
                if current_row.iloc[0] >= year_high.iloc[0] - (year_high.iloc[0] - year_low.iloc[0]) * 0.2:
                    num_highs[current_row.name] += 1
                if current_row.iloc[0] <= year_low.iloc[0] + (year_high.iloc[0] - year_low.iloc[0]) * 0.2:
                    num_lows[current_row.name] += 1
                current_row_index += 1
                if current_row_index >= len(df):
                    break
                current_row = df.iloc[current_row_index]
    df = pd.DataFrame((num_highs, num_lows), index=["Num Highs", "Num Lows"]).T
    df["Strength"] = df["Num Highs"] / (df["Num Highs"] + df["Num Lows"]) * 100
    res = pd.concat([df_strength[df_strength.index != last_strength_date], df])
    res.to_csv("repoData/strength_ratio.csv", index_label="Date")

def load_strength_breadth_data(data_dir):
    df_strength = basic_utils.get_repo_data("strength_ratio.csv")
    df_breadth = basic_utils.get_repo_data("breadth_ratio.csv")
    last_strength_date = df_strength.index[-1]
    last_breadth_date = df_breadth.index[-1]
    sp500_dir = data_dir + "sp500/"

    # Load ticker data
    print("Loading S&P 500 data...")
    load_sp500_data(data_dir, start=min(last_strength_date, last_breadth_date))
    print("Updating date count...")
    update_date_count(data_dir)

    trading_days = pd.read_csv(data_dir + "trading_days.csv", index_col=0, header=[0,1])
    
    print("Updating breadth data...")
    update_breadth_data(sp500_dir, df_breadth, trading_days)
    print("Updating strength data...")
    update_strength_data(sp500_dir, df_strength, trading_days)

def fetch_all(data_dir, start_date, end_date=pd.Timestamp.today().date()):
    # Scrape all data
    print("Scraping AAII Sentiment Survey data...")
    scrape_aaii()
    print("Scraping Insider Transactions data...")
    scrape_insider()
    print("Scraping Margin Stats data...")
    scrape_margin_stats()
    print("Scraping Put/Call Ratio data...")
    scrape_put_call()
    print("Scraping S&P 500 companies data...")
    scrape_companies()

    load_strength_breadth_data(data_dir)
    
    print("Fetching metrics data...")
    Metric.setPreferences(data_dir, start_date, end_date)
    metrics = []
    metrics.append(T10YearYield())
    metrics.append(JunkBondSpread())
    metrics.append(SaveHavenDemand())
    metrics.append(ConsumerSentiment())
    metrics.append(SP500Momentum())
    metrics.append(PutCallRatio())
    metrics.append(InsiderTransactions())
    metrics.append(AAIISentiment())
    metrics.append(MarginStats())
    metrics.append(VIX())
    metrics.append(StockPriceBreadth())
    metrics.append(StockPriceStrength())
    metrics.append(YieldCurve())
    for metric in metrics:
        metric.get()
        metric.save()
    return metrics
