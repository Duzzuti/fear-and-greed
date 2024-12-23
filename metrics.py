from metric_base import Metric
import basic_utils
import numpy as np
import pandas as pd

class SP500Momentum(Metric):
    def __init__(self, moving_avg_window=125, shift=pd.DateOffset(days=0)):
        super().__init__(shift)
        self.moving_avg_window = moving_avg_window

    def fetch(self):
        # Fetch the S&P 500 data
        self.data = basic_utils.fetch_yf_data("^GSPC", self.data_dir, self.start_date, self.end_date, trading_days=self.trading_days)["Close"]

    def calculate(self):
        # Calculate the momentum
        self.processed = basic_utils.pct_difference_to_ema(self.data, steepness=15, window=self.moving_avg_window)
    
    def normalize(self):
        # no normalization needed
        self.result = self.processed

class VIX(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Fetch the VIX data
        self.data = basic_utils.fetch_yf_data("^VIX", self.data_dir, self.start_date, self.end_date, trading_days=self.trading_days)["Close"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.difference_to_ema(self.processed, reverse=True, steepness=0.2)

class MarginStats(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the margin stats data
        self.data = basic_utils.get_repo_data("margin_stats.csv", self.start_date, self.end_date)["Leverage Ratio"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        self.result = (basic_utils.difference_to_ema(self.processed, steepness=5, window=36) + basic_utils.normalize_tanh(self.processed, steepness=2, shift=-1.5)) / 2

class AAIISentiment(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the AAII sentiment data
        self.data = basic_utils.get_repo_data("aaii_sentiment.csv", self.start_date, self.end_date)["Bull-Bear Spread"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        self.result = basic_utils.normalize_tanh(self.processed, steepness=5)

class InsiderTransactions(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the insider transactions data
        self.data = basic_utils.get_repo_data("insider.csv", self.start_date, self.end_date)["Value"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.normalize_tanh(self.processed, steepness=7, shift=0.4, reverse=True)

class PutCallRatio(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the put-call ratio data
        self.data = basic_utils.get_repo_data("put_call_ratios.csv", self.start_date, self.end_date)

    def calculate(self):
        # ffill the zeros
        self.processed = self.data.replace(0, np.nan)
        self.processed.ffill(inplace=True)

    def normalize(self):
        # Normalize the data
        self.result = basic_utils.difference_to_ema(self.processed, steepness=15, reverse=True, window=500).ewm(span=5).mean()

class ConsumerSentiment(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Load the consumer sentiment data
        self.data = basic_utils.fetch_fred_data("UMCSENT", self.data_dir, self.start_date, self.end_date)

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.difference_to_ema(self.processed, steepness=0.2, window=36)

class SaveHavenDemand(Metric):
    def __init__(self, period=100, bond_weight=None, shift=pd.DateOffset(days=0)):
        super().__init__(shift)
        self.period = period
        self.bond_weight = bond_weight

    def fetch(self):
        # Load the safe haven demand data
        self.data = basic_utils.fetch_yf_data("^SP500TR", self.data_dir, self.start_date, self.end_date, trading_days=self.trading_days)["Close"]
        self.tnx = basic_utils.fetch_yf_data("^TNX", self.data_dir, self.start_date, self.end_date, trading_days=self.trading_days)["Close"]

    def calculate(self):
        # calculate the period returns of the stock market
        sp500_period_return = self.data.pct_change(self.period)
        # calculate the return per annum
        sp500_annual_return = ((1 + sp500_period_return) ** (252 / self.period) - 1) * 100
        # calculate the difference between the first columns
        bond_weight = self.bond_weight
        if bond_weight == None:
            bond_weight = 252 / self.period
        sp500_annual_return["Diff"] = sp500_annual_return["^SP500TR"] - self.tnx["^TNX"] * bond_weight
        self.processed = sp500_annual_return[["Diff"]]
    
    def normalize(self):
        self.result = basic_utils.normalize_tanh(self.processed, steepness=0.05)

class JunkBondSpread(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Load the junk bond spread data
        self.data = basic_utils.fetch_fred_data("BAMLH0A0HYM2", self.data_dir, self.start_date, self.end_date)
        self.data.ffill(inplace=True)

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.difference_to_ema(self.processed, steepness=1, reverse=True, window=252)

class YieldCurve(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Load the yield curve data
        self.data = basic_utils.fetch_fred_data("T10Y2Y", self.data_dir, self.start_date, self.end_date)
        self.data.ffill(inplace=True)

    def calculate(self):
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.normalize_tanh(self.processed, shift=1, reverse=True)

class T10YearYield(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Load the 10 year yield data
        self.data = basic_utils.fetch_yf_data("^TNX", self.data_dir, self.start_date, self.end_date, trading_days=self.trading_days)["Close"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.difference_to_ema(self.processed, steepness=2, window=500)
    
class StockPriceBreadth(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Load the stock price breadth data
        self.data = basic_utils.get_repo_data("breadth_ratio.csv", self.start_date, self.end_date)["Breadth Ratio"]

    def calculate(self):
        # flatten the data
        self.processed = self.data.ewm(span=100, min_periods=50).mean() * 100
    
    def normalize(self):
        # Normalize the data
        self.result = basic_utils.normalize_tanh(self.processed, steepness=0.35, shift=-52)

class StockPriceStrength(Metric):
    def __init__(self, shift=pd.DateOffset(days=0)):
        super().__init__(shift)

    def fetch(self):
        # Load the stock price strength data
        self.data = basic_utils.get_repo_data("strength_ratio.csv", self.start_date, self.end_date)["Strength"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # No normalization needed
        self.result = self.processed