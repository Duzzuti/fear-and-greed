from metric_base import Metric
import utils

class SP500Momentum(Metric):
    def __init__(self, moving_avg_window=125):
        super().__init__()
        self.moving_avg_window = moving_avg_window

    def fetch(self):
        # Fetch the S&P 500 data
        self.data = utils.fetch_yf_data("^GSPC", self.data_dir, self.start_date, self.end_date)["Close"]

    def calculate(self):
        # Calculate the momentum
        moving_avg = self.data.rolling(window=self.moving_avg_window).mean()
        momentum = (self.data - moving_avg) / moving_avg
        self.processed = momentum * 100  # Convert to percentage
    
    def normalize(self):
        # Normalize the data
        self.result = utils.difference_to_ema(self.processed, steepness=0.2)

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