from metric_base import Metric
import utils
import numpy as np

class SP500Momentum(Metric):
    def __init__(self, moving_avg_window=125):
        super().__init__()
        self.moving_avg_window = moving_avg_window

    def fetch(self):
        # Fetch the S&P 500 data
        self.data = utils.fetch_yf_data("^GSPC", self.data_dir, self.start_date, self.end_date)["Close"]

    def calculate(self):
        # Calculate the momentum
        self.processed = utils.pct_difference_to_ema(self.data, steepness=15, window=self.moving_avg_window)
    
    def normalize(self):
        # no normalization needed
        self.result = self.processed

class VIX(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Fetch the VIX data
        self.data = utils.fetch_yf_data("^VIX", self.data_dir, self.start_date, self.end_date)["Close"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = utils.difference_to_ema(self.processed, reverse=True, steepness=0.2)

class MarginStats(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the margin stats data
        self.data = utils.get_repo_data("margin_stats.csv", self.start_date, self.end_date)["Leverage Ratio"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # no normalization needed
        self.result = self.processed

class AAIISentiment(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the AAII sentiment data
        self.data = utils.get_repo_data("aaii_sentiment.csv", self.start_date, self.end_date)["Bull-Bear Spread"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # no normalization needed
        self.result = self.processed

class InsiderTransactions(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the insider transactions data
        self.data = utils.get_repo_data("insider.csv", self.start_date, self.end_date)["Value"]

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = utils.difference_to_ema(self.processed, window=5, steepness=10)

class PutCallRatio(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the put-call ratio data
        self.data = utils.get_repo_data("put_call_ratios.csv", self.start_date, self.end_date)["PCR"]

    def calculate(self):
        # ffill the zeros
        self.processed = self.data.replace(0, np.nan)
        self.processed.ffill(inplace=True)

    def normalize(self):
        # Normalize the data
        self.result = utils.difference_to_ema(self.processed, steepness=10, reverse=True).ewm(span=3).mean()

class ConsumerSentiment(Metric):
    def __init__(self):
        super().__init__()

    def fetch(self):
        # Load the consumer sentiment data
        self.data = utils.fetch_fred_data("UMCSENT", self.data_dir, self.start_date, self.end_date)

    def calculate(self):
        # no calculation needed
        self.processed = self.data
    
    def normalize(self):
        # Normalize the data
        self.result = utils.difference_to_ema(self.processed, steepness=0.2, window=24)